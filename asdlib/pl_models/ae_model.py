from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import Tensor

from ..models.audio_feature.base import BaseAudioFeature
from ..utils.config_class import GradConfig
from ..utils.config_class.output_config import PLOutput
from ..utils.pl_utils import instantiate_tgt
from .base_model import BasePLModel


class AEPLModel(BasePLModel):
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

    def _constructor(
        self,
        network_cfg: Dict[str, Any] = {},
        audio_feat_cfg: Dict[str, Any] = {},
        loss_cfg: Dict[str, Any] = {},
        use_compile: bool = False,
        condition_label: Optional[str] = None,
    ) -> None:
        self.audio_feat: BaseAudioFeature = instantiate_tgt(audio_feat_cfg)
        self.network = instantiate_tgt(
            {"input_dim": self.audio_feat.feat_dim, **network_cfg}
        )
        if use_compile:
            self.network = torch.compile(self.network)  # type: ignore
        self.loss = instantiate_tgt(loss_cfg)
        self.condition_label = condition_label

    def forward(self, batch: dict) -> PLOutput:
        x_ref = self.audio_feat(batch["wave"])
        x_est, z = self.network(x_ref)
        embed_dict = {"main": z}
        anomaly_score_dict = {"plain": self.loss(x_est, x_ref)}
        return PLOutput(embed=embed_dict, AS=anomaly_score_dict)

    def wave2loss(self, wave: Tensor, batch: Dict[str, Tensor]) -> Dict[str, Any]:
        x_ref = self.audio_feat(wave)
        if self.condition_label is None:
            condition = None
        else:
            condition = batch[self.condition_label]
        x_est, z = self.network(x_ref, condition=condition)
        loss_dict = {"main": self.loss(x_est, x_ref)}
        return loss_dict

    def training_step(self, batch, batch_idx):
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
