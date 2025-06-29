import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from asdit.utils.common import instantiate_tgt
from asdit.utils.config_class import GradConfig

from .base_plmodel import BasePLFrontend

logger = logging.getLogger(__name__)


class BasicDisPLModel(BasePLFrontend):
    def __init__(
        self,
        model_cfg: Dict[str, Any],
        optim_cfg: Dict[str, Any],
        grad_cfg: GradConfig,
        lrscheduler_cfg: Optional[Dict[str, Any]] = None,
        label_dict_path: Optional[Dict[str, Path]] = None,
    ):
        super().__init__(
            model_cfg=model_cfg,
            optim_cfg=optim_cfg,
            grad_cfg=grad_cfg,
            lrscheduler_cfg=lrscheduler_cfg,
            label_dict_path=label_dict_path,
        )

    def set_head_dict(self, label_to_lossweight_dict: Dict[str, float]):
        self.head_dict = torch.nn.ModuleDict({})
        for label_name in label_to_lossweight_dict:
            loss_cfg = {
                "n_classes": self.num_class_dict[label_name],
                "embed_size": self.embed_size,
                **self.loss_cfg,
            }
            self.head_dict[label_name] = instantiate_tgt(loss_cfg)

    def set_augmentations(self, augmentation_cfg_list: List[Dict[str, Any]]):
        self.augmentations = torch.nn.ModuleList([])
        for cfg in augmentation_cfg_list:
            self.augmentations.append(instantiate_tgt(cfg))

    def construct_model(
        self,
        extractor_cfg: Dict[str, Any],
        loss_cfg: Dict[str, Any],
        label_to_lossweight_dict: Dict[str, float],
        augmentation_cfg_list: Optional[List[Dict[str, Any]]] = None,
        use_compile: bool = False,
    ) -> None:
        if augmentation_cfg_list is None:
            augmentation_cfg_list = []
        self.extractor = instantiate_tgt(extractor_cfg)
        if use_compile:
            self.extractor = torch.compile(self.extractor)  # type: ignore
        self.loss_cfg = loss_cfg
        self.embed_size = self.extractor.embed_size
        self.label_to_lossweight_dict = label_to_lossweight_dict
        self.set_head_dict(self.label_to_lossweight_dict)
        self.set_augmentations(augmentation_cfg_list)

    def forward(self, batch: dict) -> Dict[str, Any]:
        embed = self.extractor(batch["wave"])  # (B, D)
        return {"embed": embed}

    def wave2loss(self, wave: Tensor, batch: Dict[str, Tensor]) -> Dict[str, Any]:
        embed = self.extractor(wave)
        loss_dict = {"main": 0.0}

        for label_name, weight in self.label_to_lossweight_dict.items():
            loss_dict[label_name] = self.head_dict[label_name](
                embed, batch[f"onehot_{label_name}"]
            )
            loss_dict["main"] += loss_dict[label_name] * weight
        return loss_dict

    def training_step(self, batch, batch_idx):
        for aug_func in self.augmentations:
            batch = aug_func(batch)
        wave = batch.pop("wave")
        loss_dict = self.wave2loss(wave, batch)
        self.log_loss(torch.tensor(len(wave)).float(), "train/batch_size", 1)
        for key_, val_ in loss_dict.items():
            self.log_loss(val_, f"train/{key_}", len(wave))
        return loss_dict["main"]

    def validation_step(self, batch, batch_idx):
        wave = batch.pop("wave")
        loss_dict = self.wave2loss(wave, batch)
        self.log_loss(torch.tensor(len(wave)).float(), "valid/batch_size", 1)
        for key_, val_ in loss_dict.items():
            self.log_loss(val_, f"valid/{key_}", len(wave))
