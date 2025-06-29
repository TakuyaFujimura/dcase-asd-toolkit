import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import lightning.pytorch as pl
import numpy as np
import torch

from asdit.utils.common import instantiate_tgt
from asdit.utils.dcase_utils import get_label_dict
from asdit.utils.dcase_utils.dcase_idx import get_domain_idx

from .auroc import AUROC
from .base import BaseFrontend

logger = logging.getLogger(__name__)


def grad_norm(module: torch.nn.Module) -> float:
    total_norm = 0
    for p in module.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)  # type: ignore
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


class BasePLFrontend(pl.LightningModule, BaseFrontend):
    def __init__(
        self,
        model_cfg: Dict[str, Any],
        optim_cfg: Dict[str, Any],
        lrscheduler_cfg: Optional[Dict[str, Any]] = None,
        label_dict_path: Optional[Dict[str, Path]] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.optim_cfg = optim_cfg
        self.lrscheduler_cfg = lrscheduler_cfg
        self.num_class_dict: Dict[str, int] = {}
        if label_dict_path is None:
            label_dict_path = {}
        for key_, val_ in get_label_dict(label_dict_path).items():
            self.num_class_dict[key_] = val_.num_class

        self.construct_model(**model_cfg)

    def construct_model(self, *args, **kwargs):
        pass

    def extract(self, batch: dict) -> Dict[str, Any]:
        return self(batch)

    def log_loss(self, loss: Any, log_name: str, batch_size: int):
        if isinstance(loss, torch.Tensor) and loss.numel() == 1:
            self.log(
                log_name,
                loss,
                prog_bar=True,
                batch_size=batch_size,
                sync_dist=True,
                on_step=True,
                on_epoch=False,
            )
        else:
            raise ValueError("Loss is not a scalar tensor")

    def on_after_backward(self):
        grad_norm_val = grad_norm(self)
        opt = self.trainer.optimizers[0]
        current_lr = opt.state_dict()["param_groups"][0]["lr"]
        self.logger.log_metrics(  # type: ignore
            {
                "grad/norm": grad_norm_val,
                "grad/lr": current_lr,
                "grad/step_size": current_lr * grad_norm_val,
            },
            step=self.trainer.global_step,
        )

    def configure_optimizers(self):
        optimizer = instantiate_tgt({"params": self.parameters(), **self.optim_cfg})
        if self.lrscheduler_cfg is not None:
            scheduler = instantiate_tgt(
                {"optimizer": optimizer, **self.lrscheduler_cfg}
            )
            lr_scheduler = {"scheduler": scheduler, "interval": "step"}
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        else:
            return optimizer

    def lr_scheduler_step(self, scheduler: Any, metric: Optional[Any]) -> None:
        if scheduler.__class__.__module__.startswith("timm.scheduler."):
            scheduler.step(self.global_step)
        else:
            super().lr_scheduler_step(scheduler=scheduler, metric=metric)


class BasePLAUCFrontend(BasePLFrontend):
    def __init__(
        self,
        model_cfg: Dict[str, Any],
        optim_cfg: Dict[str, Any],
        lrscheduler_cfg: Optional[Dict[str, Any]] = None,
        label_dict_path: Optional[Dict[str, Path]] = None,
    ):
        super().__init__(
            model_cfg=model_cfg,
            optim_cfg=optim_cfg,
            lrscheduler_cfg=lrscheduler_cfg,
            label_dict_path=label_dict_path,
        )
        self.anomaly_score_name: Optional[List[str]] = None

    def setup_auc(self):
        if self.anomaly_score_name is None:
            raise ValueError(
                "Please set self.anomaly_score_name before setup_auc and after __init__()"
            )
        self.auc_type_list = [
            "all_s_auc",  # AUC with all sections in source domain
            "all_t_auc",
            "all_smix_auc",
            "all_tmix_auc",
            "all_mix_auc",
        ]
        self.auroc_model_dict: Dict[str, AUROC] = {}
        for auc_key in self.auc_type_list:
            for as_key in self.anomaly_score_name:
                self.auroc_model_dict[f"{auc_key}_{as_key}"] = AUROC()

    def validation_step_auc(
        self, anomaly_score_dict: Dict[str, torch.Tensor], batch: dict
    ):
        is_normal_np = np.array(batch["is_normal"])
        is_target_np = np.array(batch["is_target"])
        is_anomaly_tensor = 1 - torch.tensor(batch["is_normal"])
        for auc_model_key in self.auroc_model_dict:
            auc_type = "_".join(auc_model_key.split("_")[:3])
            as_key = "_".join(auc_model_key.split("_")[3:])
            domain_idx_np = get_domain_idx(
                auc_type=auc_type,
                is_target=is_target_np,
                is_normal=is_normal_np,
            )
            domain_idx = torch.tensor(domain_idx_np, dtype=torch.bool)
            score_tensor = anomaly_score_dict[as_key]
            self.auroc_model_dict[auc_model_key].update(
                score=score_tensor[domain_idx],
                target=is_anomaly_tensor[domain_idx],
            )

    def on_validation_epoch_end(self) -> None:
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
