from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from asdkit.utils.common.instantiate_util import instantiate_tgt

from .discriminative_model import BasicDisPLModel


class SubspaceLossPLModel(BasicDisPLModel):
    def __init__(
        self,
        model_cfg: Dict[str, Any],
        optim_cfg: Dict[str, Any],
        lrscheduler_cfg: Optional[Dict[str, Any]] = None,
        label_dict_path: Optional[Dict[str, Path]] = None,
        save_only_trainable: bool = False,
    ):
        """
        T. Fujimura et al., "Improvements of discriminative feature space training for anomalous sound detection in unlabeled conditions," Proc. ICASSP, 2025.

        Args:
            model_cfg (Dict[str, Any]): Configuration for the model. Parameters in this dictionary are used in `self.construct_model`.
            optim_cfg (Dict[str, Any]): Configuration for the optimizer.
            lrscheduler_cfg (Optional[Dict[str, Any]]): Configuration for the learning rate scheduler.
            label_dict_path (Optional[Dict[str, Path]]): Dictionary for label_dict paths.
            save_only_trainable (bool): If True, only trainable parameters are saved.
        """
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
        self.head_dict = torch.nn.ModuleDict({})
        self.subspace_head_dict = torch.nn.ModuleDict({})
        for label_name in label_to_lossweight_dict:
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
        extractor_cfg: Dict[str, Any],
        loss_cfg: Dict[str, Any],
        label_to_lossweight_dict: Dict[str, float],
        augmentation_cfg_list: Optional[List[Dict[str, Any]]] = None,
        use_compile: bool = False,
        subspace_embed_size: Optional[int] = None,
        subspace_loss_weight: float = 1.0,
    ) -> None:
        if subspace_embed_size is None:
            raise ValueError("subspace_embed_size is should be specified")
        self.subspace_embed_size = subspace_embed_size
        self.subspace_loss_weight = subspace_loss_weight
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
        loss_dict = {"main": 0.0}

        for label_name, weight in self.label_to_lossweight_dict.items():
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
                + self.subspace_loss_weight * loss_subspace
            )
            loss_dict["main"] += loss_dict[label_name] * weight

        return loss_dict
