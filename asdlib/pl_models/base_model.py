from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import lightning.pytorch as pl
import torch

from asdlib.utils.common import instantiate_tgt
from asdlib.utils.config_class import GradConfig, PLOutput
from asdlib.utils.dcase_utils import get_label_dict


def grad_norm(module: torch.nn.Module) -> float:
    total_norm = 0
    for p in module.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)  # type: ignore
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    return total_norm


class BasePLModel(pl.LightningModule, ABC):
    def __init__(
        self,
        model_cfg: Dict[str, Any],
        optim_cfg: Dict[str, Any],
        scheduler_cfg: Optional[Dict[str, Any]],
        grad_cfg: GradConfig,
        label_dict_path: Dict[str, Path],  # given by config.label_dict_path in train.py
    ):
        super().__init__()
        self.save_hyperparameters()
        self.optim_cfg = optim_cfg
        self.scheduler_cfg = scheduler_cfg
        self.grad_cfg = grad_cfg
        self.num_class_dict: Dict[str, int] = {}
        for key_, val_ in get_label_dict(label_dict_path).items():
            self.num_class_dict[key_] = val_.num_class

        if self.grad_cfg.clipper_cfg is not None:
            self.grad_clipper = instantiate_tgt(self.grad_cfg.clipper_cfg)
        else:
            self.grad_clipper = None

        self._constructor(**model_cfg)

    def _constructor(self):
        pass

    @abstractmethod
    def forward(self, batch: dict) -> PLOutput:
        pass

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
        if self.grad_clipper is not None:
            grad_norm_val, clipping_threshold = self.grad_clipper(self)
            clipped_norm_val = min(grad_norm_val, clipping_threshold)
        else:
            grad_norm_val = grad_norm(self)
            clipped_norm_val = grad_norm_val

        if self.trainer.global_step % self.grad_cfg.log_every_n_steps == 0:
            opt = self.trainer.optimizers[0]
            current_lr = opt.state_dict()["param_groups"][0]["lr"]
            self.logger.log_metrics(
                {
                    "grad/norm": grad_norm_val,
                    "grad/clipped_norm": clipped_norm_val,
                    "grad/lr": current_lr,
                    "grad/step_size": current_lr * clipped_norm_val,
                },
                step=self.trainer.global_step,
            )

    def configure_optimizers(self):
        optimizer = instantiate_tgt({"params": self.parameters(), **self.optim_cfg})
        if self.scheduler_cfg is not None:
            scheduler = instantiate_tgt({"optimizer": optimizer, **self.scheduler_cfg})
            lr_scheduler = {"scheduler": scheduler, "interval": "step"}
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        else:
            return optimizer

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step()
