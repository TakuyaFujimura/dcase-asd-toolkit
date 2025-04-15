from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor

from asdit.models.audio_feature.base import BaseAudioFeature
from asdit.utils.common import instantiate_tgt
from asdit.utils.config_class import GradConfig, PLOutput

from .base_plmodel import BasePLAUCFrontend


class AEPLModel(BasePLAUCFrontend):
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

    def construct_model(
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
        self.loss = instantiate_tgt({**loss_cfg, "reduction": "none"})
        self.condition_label = condition_label
        self.anomaly_score_name = ["recon"]
        self.setup_auc()

    def feat2net(
        self, x_ref: Tensor, batch: Dict[str, Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.condition_label is None:
            condition = None
        else:
            condition = batch[self.condition_label]
        x_est, z = self.network(x_ref, condition=condition)
        return x_est, z

    def net2as(
        self, x_ref: torch.Tensor, x_est: torch.Tensor, z: torch.Tensor, n_sample: int
    ) -> Dict[str, torch.Tensor]:
        recon_loss = self.loss(x_est, x_ref).view(n_sample, -1).mean(dim=1)
        return {"recon": recon_loss}

    def forward(self, batch: dict) -> PLOutput:
        wave = batch["wave"]
        x_ref = self.audio_feat(wave)
        x_est, z = self.feat2net(x_ref=x_ref, batch=batch)

        z = z.view(len(wave), -1, z.shape[-1]).mean(dim=1)
        # (B, n_frames, z_dim) -> (B, z_dim), time average

        embed_dict = {"main": z}
        anomaly_score_dict = self.net2as(
            x_ref=x_ref, x_est=x_est, z=z, n_sample=len(wave)
        )
        return PLOutput(embed=embed_dict, AS=anomaly_score_dict)

    def training_step(self, batch, batch_idx):
        x_ref = batch["feat"]
        x_est, z = self.feat2net(x_ref=x_ref, batch=batch)
        loss_dict = {"main": self.loss(x_est, x_ref).mean()}

        self.log_loss(torch.tensor(len(x_ref)).float(), "train/batch_size", 1)
        for key_, val_ in loss_dict.items():
            self.log_loss(val_, f"train/{key_}", len(x_ref))
        return loss_dict["main"]

    def validation_step(self, batch, batch_idx):
        wave = batch["wave"]
        x_ref = self.audio_feat(wave)
        x_est, z = self.feat2net(x_ref=x_ref, batch=batch)
        loss_dict = {"main": self.loss(x_est, x_ref).mean()}
        anomaly_score_dict = self.net2as(
            x_ref=x_ref, x_est=x_est, z=z, n_sample=len(wave)
        )

        self.log_loss(torch.tensor(len(wave)).float(), "valid/batch_size", 1)
        for key_, val_ in loss_dict.items():
            self.log_loss(val_, f"valid/{key_}", len(wave))
        self.validation_step_auc(anomaly_score_dict=anomaly_score_dict, batch=batch)
