from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor

from ..models.audio_feature.base import BaseAudioFeature
from ..utils.config_class import GradConfig
from ..utils.config_class.output_config import PLOutput
from ..utils.dcase_utils import get_domain_idx
from ..utils.pl_utils import instantiate_tgt
from .auroc import AUROC
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

    def setup_auc(self):
        self.auc_type_list = [
            "all_s_auc",
            "all_t_auc",
            "all_smix_auc",
            "all_tmix_auc",
            "all_mix_auc",
        ]
        self.anomaly_score_name = ["recon"]
        self.auroc_model_dict: Dict[str, AUROC] = {}
        for auc_key in self.auc_type_list:
            for as_key in self.anomaly_score_name:
                self.auroc_model_dict[f"{auc_key}_{as_key}"] = AUROC()
        self.auroc_data_dict = {}

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
        self.loss = instantiate_tgt({**loss_cfg, "reduction": "none"})
        self.condition_label = condition_label
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
        self.validation_auc(anomaly_score_dict=anomaly_score_dict, batch=batch)

    def validation_auc(self, anomaly_score_dict: Dict[str, torch.Tensor], batch: dict):
        is_normal_np = np.array(batch["is_normal"])
        is_target_np = np.array(batch["is_target"])
        is_anomaly_tensor = 1 - torch.tensor(
            batch["is_normal"], dtype=torch.int32, device="cpu"
        )
        for auc_model_key in self.auroc_model_dict:
            auc_type = "_".join(auc_model_key.split("_")[:3])
            as_key = "_".join(auc_model_key.split("_")[3:])

            domain_idx_np = get_domain_idx(
                auc_type=auc_type,
                is_target=is_target_np,
                is_normal=is_normal_np,
            )
            domain_idx = torch.tensor(domain_idx_np, dtype=torch.bool)

            score_tensor = anomaly_score_dict[as_key].detach().cpu()
            # TorchMetricsUserWarning: You are trying to use a metric in deterministic mode on GPU
            # that uses `torch.cumsum`, which is currently not supported. The tensor will be copied
            # to the CPU memory to compute it and then copied back to GPU. Expect some slowdowns.

            self.auroc_model_dict[auc_model_key].update(
                score=score_tensor[domain_idx],
                target=is_anomaly_tensor[domain_idx],
            )

            if auc_model_key in self.auroc_data_dict:
                self.auroc_data_dict[auc_model_key]["preds"] = torch.cat(
                    [
                        self.auroc_data_dict[auc_model_key]["preds"],
                        score_tensor[domain_idx],
                    ]
                )
                self.auroc_data_dict[auc_model_key]["target"] = torch.cat(
                    [
                        self.auroc_data_dict[auc_model_key]["target"],
                        is_anomaly_tensor[domain_idx],
                    ]
                )
            else:
                self.auroc_data_dict[auc_model_key] = {
                    "preds": score_tensor[domain_idx],
                    "target": is_anomaly_tensor[domain_idx],
                }

    def on_validation_epoch_end(self) -> None:
        for auc_model_key in self.auroc_model_dict:
            auc = roc_auc_score(
                y_true=self.auroc_data_dict[auc_model_key]["target"].numpy(),
                y_score=self.auroc_data_dict[auc_model_key]["preds"].numpy(),
            )
            self.log(
                f"valid/auc/{auc_model_key}_my",
                torch.tensor([auc]),
                on_step=False,
                on_epoch=True,
            )
            print(f"valid/auc/{auc_model_key}_my: {auc}")
            auc = self.auroc_model_dict[auc_model_key].compute()
            self.log(
                f"valid/auc/{auc_model_key}",
                torch.tensor([auc]),
                on_step=False,
                on_epoch=True,
            )
            print(f"valid/auc/{auc_model_key}: {auc}")
            self.auroc_model_dict[auc_model_key].reset()

        self.auroc_data_dict = {}
