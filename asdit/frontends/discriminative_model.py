import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from asdit.utils.common import instantiate_tgt
from asdit.utils.config_class import GradConfig, PLOutput

from .base_plmodel import BasePLFrontend

logger = logging.getLogger(__name__)


class BasicDisPLModel(BasePLFrontend):
    def __init__(
        self,
        model_cfg: Dict[str, Any],
        optim_cfg: Dict[str, Any],
        scheduler_cfg: Optional[Dict[str, Any]],
        grad_cfg: GradConfig,
        label_dict_path: Dict[str, Path],  # given by config.label_dict_path in train.py
        partially_saved_param_list: List[str] = [],
    ):
        super().__init__(
            model_cfg=model_cfg,
            optim_cfg=optim_cfg,
            scheduler_cfg=scheduler_cfg,
            grad_cfg=grad_cfg,
            label_dict_path=label_dict_path,
            partially_saved_param_list=partially_saved_param_list,
        )

    def check_loss_cfg(self, loss_cfg: Dict[str, Any]):
        # If tgt_class is known loss class, this check the normalize flag.
        # e.g., tgt_class: asdit.models.losses.SCAdaCos
        split_loss_tgt = loss_cfg["tgt_class"].split(".")
        if "asdit.losses" != ".".join(split_loss_tgt[:-1]):
            return 0

        if split_loss_tgt[-1] not in ["AdaCos", "ArcFace", "SCAdaCos", "AdaProj"]:
            return 0

        if not self.normalize:
            raise ValueError("normalize is False, but loss uses normalized embedding.")
        else:
            return 0

    def set_head_dict(self, label_to_lossratio_dict: Dict[str, float]):
        self.head_dict = torch.nn.ModuleDict({})
        for label_name in label_to_lossratio_dict:
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
        normalize: bool,
        extractor_cfg: Dict[str, Any] = {},
        loss_cfg: Dict[str, Any] = {},
        label_to_lossratio_dict: Dict[str, float] = {},
        augmentation_cfg_list: List[Dict[str, Any]] = [],
        use_compile: bool = False,
    ) -> None:
        self.normalize = normalize
        self.extractor = instantiate_tgt(extractor_cfg)
        if use_compile:
            self.extractor = torch.compile(self.extractor)  # type: ignore
        self.loss_cfg = loss_cfg
        self.embed_size = self.extractor.embed_size
        self.label_to_lossratio_dict = label_to_lossratio_dict
        self.check_loss_cfg(self.loss_cfg)
        self.set_head_dict(self.label_to_lossratio_dict)
        self.set_augmentations(augmentation_cfg_list)

    def forward(self, batch: dict) -> PLOutput:
        logit_dict: Dict[str, Tensor] = {}
        embed = self.extractor(batch["wave"])  # (B, D)

        for label_name in self.label_to_lossratio_dict:
            logits = self.head_dict[label_name].calc_logits(embed)  # type: ignore
            logit_dict[label_name] = logits

        if self.normalize:
            embed_dict = {"main": F.normalize(embed, p=2, dim=1)}
        else:
            embed_dict = {"main": embed}

        return PLOutput(embed=embed_dict, logits=logit_dict)

    def wave2loss(self, wave: Tensor, batch: Dict[str, Tensor]) -> Dict[str, Any]:
        embed = self.extractor(wave)
        loss_dict = {"main": 0.0}

        for label_name, ratio in self.label_to_lossratio_dict.items():
            loss_dict[label_name] = self.head_dict[label_name](
                embed, batch[f"onehot_{label_name}"]
            )
            loss_dict["main"] += loss_dict[label_name] * ratio
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
