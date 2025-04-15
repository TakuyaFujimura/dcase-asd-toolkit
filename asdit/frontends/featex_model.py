from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from asdit.augmentations.featex import FeatEx
from asdit.utils.common.instantiate_util import instantiate_tgt
from asdit.utils.config_class.train_config import GradConfig

from .discriminative_model import BasicDisPLModel


class FeatExPLModel(BasicDisPLModel):
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

    def set_head_dict(self, label_to_lossratio_dict: Dict[str, float]):
        if self.embed_size % self.subspace_embed_size != 0:
            raise ValueError(
                f"embed_size {self.embed_size} should be divisible by subspace_embed_size {self.subspace_embed_size}"
            )
        embed_num = self.embed_size // self.subspace_embed_size
        self.head_dict = torch.nn.ModuleDict({})
        self.featex_head_dict = torch.nn.ModuleDict({})
        for label_name in label_to_lossratio_dict:
            main_loss_cfg = {
                "n_classes": self.num_class_dict[label_name],
                "embed_size": self.embed_size,
                **self.loss_cfg,
            }
            self.head_dict[label_name] = instantiate_tgt(main_loss_cfg)
            featex_loss_cfg = {
                "n_classes": self.num_class_dict[label_name] * (embed_num + 1),
                "embed_size": self.embed_size,
                **self.loss_cfg,
                "trainable": True,
            }
            self.featex_head_dict[label_name] = instantiate_tgt(featex_loss_cfg)

    def construct_model(  # type: ignore
        self,
        normalize: bool,
        extractor_cfg: Dict[str, Any] = {},
        loss_cfg: Dict[str, Any] = {},
        label_to_lossratio_dict: Dict[str, float] = {},
        augmentation_cfg_list: List[Dict[str, Any]] = [],
        use_compile: bool = False,
        subspace_embed_size: Optional[int] = None,
        featex_loss_ratio: float = 1.0,
        featex_prob: float = 0.5,
    ) -> None:
        if subspace_embed_size is None:
            raise ValueError("subspace_embed_size is should be specified")
        self.subspace_embed_size = subspace_embed_size
        self.featex_loss_ratio = featex_loss_ratio
        self.featex = FeatEx(prob=featex_prob, subspace_embed_size=subspace_embed_size)
        super().construct_model(
            normalize=normalize,
            extractor_cfg=extractor_cfg,
            loss_cfg=loss_cfg,
            label_to_lossratio_dict=label_to_lossratio_dict,
            augmentation_cfg_list=augmentation_cfg_list,
            use_compile=use_compile,
        )

    def wave2loss(self, wave: Tensor, batch: Dict[str, Tensor]) -> Dict[str, Any]:
        embed = self.extractor(wave)
        assert embed.shape[1] == self.embed_size
        embed_ex, batch_ex = self.featex(embed=embed, batch=batch)
        loss_dict = {"main": 0.0}

        for label_name, ratio in self.label_to_lossratio_dict.items():
            loss_dict[f"{label_name}_main"] = self.head_dict[label_name](
                embed, batch[f"onehot_{label_name}"]
            )
            loss_dict[f"{label_name}_featex"] = self.featex_head_dict[label_name](
                embed_ex, batch_ex[f"onehot_{label_name}"]
            )
            loss_dict[label_name] = (
                loss_dict[f"{label_name}_main"]
                + self.featex_loss_ratio * loss_dict[f"{label_name}_featex"]
            )
            loss_dict["main"] += loss_dict[label_name] * ratio

        return loss_dict
