from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from asdit.utils.common.instantiate_util import instantiate_tgt
from asdit.utils.config_class.train_config import GradConfig

from .discriminative_model import BasicDisPLModel


class SubspaceLossPLModel(BasicDisPLModel):
    def __init__(
        self,
        model_cfg: Dict[str, Any],
        optim_cfg: Dict[str, Any],
        scheduler_cfg: Optional[Dict[str, Any]],
        grad_cfg: GradConfig,
        label_dict_path: Dict[str, Path],  # given by config.label_dict_path in train.py
    ):
        super().__init__(
            model_cfg=model_cfg,
            optim_cfg=optim_cfg,
            scheduler_cfg=scheduler_cfg,
            grad_cfg=grad_cfg,
            label_dict_path=label_dict_path,
        )

    def set_head_dict(self, loss_label2lam_dict: Dict[str, float]):
        if self.embed_size % self.subspace_embed_size != 0:
            raise ValueError(
                f"embed_size {self.embed_size} should be divisible by subspace_embed_size {self.subspace_embed_size}"
            )
        self.head_dict = torch.nn.ModuleDict({})
        self.subspace_head_dict = torch.nn.ModuleDict({})
        for label_name in loss_label2lam_dict:
            main_loss_cfg = {
                "n_classes": self.num_class_dict[label_name],
                "embed_size": self.embed_size,
                **self.loss_cfg,
            }
            self.head_dict[label_name] = instantiate_tgt(main_loss_cfg)
            subspace_loss_cfg = {
                "n_classes": self.num_class_dict[label_name],
                "embed_size": self.subspace_embed_size,
                **self.loss_cfg,
                "trainable": True,
            }
            self.subspace_head_dict[label_name] = torch.nn.ModuleList(
                [
                    instantiate_tgt(subspace_loss_cfg)
                    for _ in range(self.embed_size // self.subspace_embed_size)
                ]
            )

    def construct_model(
        self,
        subspace_embed_size: int,
        normalize: bool,
        extractor_cfg: Dict[str, Any] = {},
        loss_cfg: Dict[str, Any] = {},
        loss_label2lam_dict: Dict[str, float] = {},
        augmentation_cfg_list: List[Dict[str, Any]] = [],
        use_compile: bool = False,
        subspace_loss_ratio: float = 1.0,
    ) -> None:
        self.subspace_embed_size = subspace_embed_size
        self.subspace_loss_ratio = subspace_loss_ratio
        super().construct_model(
            normalize=normalize,
            extractor_cfg=extractor_cfg,
            loss_cfg=loss_cfg,
            loss_label2lam_dict=loss_label2lam_dict,
            augmentation_cfg_list=augmentation_cfg_list,
            use_compile=use_compile,
        )

    def wave2loss(self, wave: Tensor, batch: Dict[str, Tensor]) -> Dict[str, Any]:
        embed = self.extractor(wave)
        assert embed.shape[1] == self.embed_size
        loss_dict = {"main": 0.0}

        for label_name, lam in self.loss_label2lam_dict.items():
            loss_dict[f"{label_name}_main"] = self.head_dict[label_name](
                embed, batch[f"onehot_{label_name}"]
            )
            loss_subspace = 0.0
            for i, subspace_head in enumerate(self.subspace_head_dict[label_name]):
                subspace_embed = embed[
                    :, i * self.subspace_embed_size : (i + 1) * self.subspace_embed_size
                ]
                loss_dict[f"{label_name}_subspace_{i}"] = subspace_head(
                    subspace_embed, batch[f"onehot_{label_name}"]
                )
                loss_subspace += loss_dict[f"{label_name}_subspace_{i}"]
            loss_dict[label_name] = (
                loss_dict[f"{label_name}_main"]
                + self.subspace_loss_ratio * loss_subspace
            )
            loss_dict["main"] += loss_dict[label_name] * lam

        return loss_dict
