from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Set, final

import lightning.pytorch as pl
import numpy as np
import torch

from asdkit.utils.common import instantiate
from asdkit.utils.dcase_utils import get_label_dict
from asdkit.utils.dcase_utils.evaluation import get_domain_mask

from .auroc import AUROC
from .base import BaseFrontend


@torch.no_grad()
def grad_norm(module: torch.nn.Module) -> float:
    """
    Args:
        module: Module whose gradients are inspected.

    Returns:
        Global gradient norm as a Python float.
    """
    norms = [p.grad.detach().norm(2) for p in module.parameters() if p.grad is not None]

    if not norms:
        return 0.0

    return torch.stack(norms).double().norm(2).item()


class BasePLFrontend(pl.LightningModule, BaseFrontend):
    """Initialize shared settings"""

    def __init__(
        self,
        model_cfg: Dict[str, Any],
        optim_cfg: Dict[str, Any],
        lrscheduler_cfg: Optional[Dict[str, Any]] = None,
        label_dict_path: Optional[Dict[str, Path]] = None,
        save_only_trainable: bool = False,
        optimizer_log_options: Optional[Dict[str, bool]] = None,
    ) -> None:
        """
        Args:
            model_cfg: Model construction configuration used by concrete frontends.
            optim_cfg: Optimizer configuration.
            lrscheduler_cfg: Optional learning-rate scheduler configuration.
            label_dict_path: Label names mapped to label dictionary files.
            save_only_trainable: Save trainable parameters, buffers, and additional
                state selected by the concrete frontend at construction time.
            optimizer_log_options: Enable optimizer metrics with keys lr,
                grad_norm, and step_size. All metrics default to enabled; supplied
                keys override these defaults. Norm and step size are measured
                before clipping.
        """
        super().__init__()
        self.checkpoint_frontend_cfg: Optional[Dict[str, Any]] = None
        self.optim_cfg = optim_cfg
        self.lrscheduler_cfg = lrscheduler_cfg
        self.optimizer_log_options = {"lr": True, "grad_norm": True, "step_size": True}
        if optimizer_log_options is not None:
            self.optimizer_log_options.update(optimizer_log_options)
        self.num_class_dict: Dict[str, int] = {}
        if label_dict_path is None:
            label_dict_path = {}
        for key, value in get_label_dict(label_dict_path).items():
            self.num_class_dict[key] = value.num_class

        # Save only trainable parameters and buffers
        self.save_only_trainable = save_only_trainable
        # None leaves checkpoint state unrestricted; an empty set saves no state.
        self.save_state_names: Optional[Set[str]] = None

    def collect_trainable_state_names(self) -> Set[str]:
        """Return trainable parameter and buffer names from the current modules."""
        return {
            name for name, param in self.named_parameters() if param.requires_grad
        } | {name for name, _ in self.named_buffers()}

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Return sample-aligned embeddings and anomaly scores for inference."""
        raise NotImplementedError

    def extract(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            batch: Input batch, including waveforms shaped (B, T).

        Returns:
            Subclass-defined embeddings and anomaly scores.
        """
        return self(batch)

    def log_loss(
        self,
        loss: Any,
        log_name: str,
        batch_size: int,
        sync_dist: bool = True,
        on_epoch: bool = False,
    ) -> None:
        """
        Args:
            loss: Scalar tensor to log.
            log_name: Metric name.
            batch_size: Number of samples used for aggregation.
            sync_dist: Synchronize values across distributed workers.
            on_epoch: Aggregate over the epoch instead of logging each step.
        """
        if isinstance(loss, torch.Tensor) and loss.numel() == 1:
            self.log(
                log_name,
                loss,
                prog_bar=True,
                batch_size=batch_size,
                sync_dist=sync_dist,
                on_step=not on_epoch,
                on_epoch=on_epoch,
            )
        else:
            raise ValueError("Loss is not a scalar tensor")

    def _log_step_losses(
        self, loss_dict: Dict[str, torch.Tensor], batch_size: int, stage: str
    ) -> None:
        """Log scalar objectives with common step/epoch and sample weighting rules."""
        self.log_loss(
            loss=torch.tensor(batch_size).float(),
            log_name=f"{stage}/batch_size",
            batch_size=1,
            sync_dist=False,
            on_epoch=False,
        )
        for key, value in loss_dict.items():
            self.log_loss(
                loss=value,
                log_name=f"{stage}/{key}",
                batch_size=batch_size,
                sync_dist=True,
                on_epoch=stage == "valid",
            )

    def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
        metrics: Dict[str, float] = {}
        current_lr = optimizer.param_groups[0]["lr"]
        if self.optimizer_log_options["lr"]:
            metrics["grad/lr"] = current_lr
        if (
            self.optimizer_log_options["grad_norm"]
            or self.optimizer_log_options["step_size"]
        ):
            grad_norm_val = grad_norm(self)
            if self.optimizer_log_options["grad_norm"]:
                metrics["grad/norm"] = grad_norm_val
            if self.optimizer_log_options["step_size"]:
                metrics["grad/step_size"] = current_lr * grad_norm_val
        if not metrics:
            return
        self.logger.log_metrics(metrics, step=self.trainer.global_step)  # type: ignore

    def configure_optimizers(self) -> Any:
        """
        Returns:
            Optimizer, or Lightning optimizer and scheduler configuration.
        """
        optimizer = instantiate(cfg=self.optim_cfg, params=self.parameters())
        if self.lrscheduler_cfg is not None:
            scheduler = instantiate(cfg=self.lrscheduler_cfg, optimizer=optimizer)
            lr_scheduler = {"scheduler": scheduler, "interval": "step"}
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        else:
            return optimizer

    def lr_scheduler_step(self, scheduler: Any, metric: Optional[Any]) -> None:
        if scheduler.__class__.__module__.startswith("timm.scheduler."):
            scheduler.step(self.global_step)
        else:
            super().lr_scheduler_step(scheduler=scheduler, metric=metric)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Save the original construction config and optionally filter selected state."""
        if self.checkpoint_frontend_cfg is None:
            raise ValueError("Set checkpoint_frontend_cfg before saving a frontend.")
        checkpoint["frontend_cfg"] = self.checkpoint_frontend_cfg
        if self.save_state_names is not None:
            state_dict = checkpoint["state_dict"]
            new_state_dict = {}
            for name, param in state_dict.items():
                if name in self.save_state_names:
                    new_state_dict[name] = param
            checkpoint["state_dict"] = new_state_dict


class BasePLAUCFrontend(BasePLFrontend):
    """Extend training infrastructure with per-domain validation AUC state."""

    def __init__(
        self,
        model_cfg: Dict[str, Any],
        optim_cfg: Dict[str, Any],
        lrscheduler_cfg: Optional[Dict[str, Any]] = None,
        label_dict_path: Optional[Dict[str, Path]] = None,
        save_only_trainable: bool = False,
        optimizer_log_options: Optional[Dict[str, bool]] = None,
    ) -> None:
        super().__init__(
            model_cfg=model_cfg,
            optim_cfg=optim_cfg,
            lrscheduler_cfg=lrscheduler_cfg,
            label_dict_path=label_dict_path,
            save_only_trainable=save_only_trainable,
            optimizer_log_options=optimizer_log_options,
        )
        self.auc_domains = ["s", "t", "smix", "tmix", "mix"]
        self.auroc_model_dict: Dict[str, AUROC] = defaultdict(AUROC)

    def validation_step_auc(
        self, anomaly_score_dict: Dict[str, torch.Tensor], batch: Dict[str, Any]
    ) -> None:
        """
        Args:
            anomaly_score_dict: Score names mapped to tensors shaped (B,).
            batch: Normality and target-domain flags for B samples.
        """
        is_normal_np = np.array(batch["is_normal"])
        is_target_np = np.array(batch["is_target"])
        is_anomaly_tensor = 1 - torch.tensor(batch["is_normal"])
        for as_key, score_tensor in anomaly_score_dict.items():
            score_tensor = score_tensor.detach().cpu()
            for domain in self.auc_domains:
                domain_mask_np = get_domain_mask(
                    domain=domain, is_target=is_target_np, is_normal=is_normal_np
                )
                domain_mask = torch.tensor(domain_mask_np, dtype=torch.bool)
                self.auroc_model_dict[f"all_{domain}_auc/{as_key}"].update(
                    score=score_tensor[domain_mask],
                    target=is_anomaly_tensor[domain_mask],
                )

    def on_validation_epoch_end(self) -> None:
        """Log available validation AUCs and reset their accumulators."""
        for auc_model_key in self.auroc_model_dict:
            auc = self.auroc_model_dict[auc_model_key].compute()
            if auc is not None:
                self.log(
                    f"valid/auc/{auc_model_key}",
                    torch.tensor([auc]),
                    on_step=False,
                    on_epoch=True,
                )
            self.auroc_model_dict[auc_model_key].reset()
