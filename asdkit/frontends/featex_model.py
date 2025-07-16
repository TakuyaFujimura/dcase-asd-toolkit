from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from asdkit.augmentations.featex import FeatEx
from asdkit.utils.common.instantiate_util import instantiate_tgt
from torch import Tensor

from .discriminative_model import BasicDisPLModel


class FeatExPLModel(BasicDisPLModel):
    def __init__(
        self,
        model_cfg: Dict[str, Any],
        optim_cfg: Dict[str, Any],
        lrscheduler_cfg: Optional[Dict[str, Any]] = None,
        label_dict_path: Optional[Dict[str, Path]] = None,
        save_only_trainable: bool = False,
    ):
        super().__init__(
            model_cfg=model_cfg,
            optim_cfg=optim_cfg,
            lrscheduler_cfg=lrscheduler_cfg,
            label_dict_path=label_dict_path,
            save_only_trainable=save_only_trainable,
        )

    def set_head_dict(self, label_to_lossweight_dict: Dict[str, float]):
        if self.embed_size % self.subspace_embed_size != 0:
            raise ValueError(
                f"embed_size {self.embed_size} should be divisible by subspace_embed_size {self.subspace_embed_size}"
            )
        embed_num = self.embed_size // self.subspace_embed_size
        self.head_dict = torch.nn.ModuleDict({})
        self.featex_head_dict = torch.nn.ModuleDict({})
        for label_name in label_to_lossweight_dict:
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

    def construct_model(
        self,
        extractor_cfg: Dict[str, Any],
        loss_cfg: Dict[str, Any],
        label_to_lossweight_dict: Dict[str, float],
        augmentation_cfg_list: Optional[List[Dict[str, Any]]] = None,
        use_compile: bool = False,
        subspace_embed_size: Optional[int] = None,
        featex_loss_weight: float = 1.0,
        featex_prob: float = 0.5,
    ) -> None:
        if subspace_embed_size is None:
            raise ValueError("subspace_embed_size is should be specified")
        self.subspace_embed_size = subspace_embed_size
        self.featex_loss_weight = featex_loss_weight
        self.featex = FeatEx(prob=featex_prob, subspace_embed_size=subspace_embed_size)
        super().construct_model(
            extractor_cfg=extractor_cfg,
            loss_cfg=loss_cfg,
            label_to_lossweight_dict=label_to_lossweight_dict,
            augmentation_cfg_list=augmentation_cfg_list,
            use_compile=use_compile,
        )

    def wave2loss(self, wave: Tensor, batch: Dict[str, Tensor]) -> Dict[str, Any]:
        embed = self.extractor(wave)
        assert embed.shape[1] == self.embed_size
        embed_ex, batch_ex = self.featex(embed=embed, batch=batch)
        loss_dict = {"main": 0.0}

        for label_name, weight in self.label_to_lossweight_dict.items():
            loss_dict[f"{label_name}_main"] = self.head_dict[label_name](
                embed, batch[f"onehot_{label_name}"]
            )
            loss_dict[f"{label_name}_featex"] = self.featex_head_dict[label_name](
                embed_ex, batch_ex[f"onehot_{label_name}"]
            )
            loss_dict[label_name] = (
                loss_dict[f"{label_name}_main"]
                + self.featex_loss_weight * loss_dict[f"{label_name}_featex"]
            )
            loss_dict["main"] += loss_dict[label_name] * weight

        return loss_dict
