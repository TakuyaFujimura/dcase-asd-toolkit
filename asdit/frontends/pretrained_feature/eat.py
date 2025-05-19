import logging
from typing import Literal, Optional

from torch import nn

from asdit.models.eat import calc_target_length, preprocess, restore
from asdit.utils.config_class.output_config import PLOutput

from .base import BaseFrozenModel

logger = logging.getLogger(__name__)


class EATFrozenModel(BaseFrozenModel):
    def __init__(self, model_cfg: Optional[dict] = None):
        super().__init__(model_cfg=model_cfg)

    def construct_model(
        self,
        ckpt_path: str,
        sec: float,
        sr: int = 16000,
        update_cfg: Optional[dict] = None,
        prediction_mode: Literal["cls", "seq"] = "cls",
    ) -> nn.Module:
        if sr != 16000:
            raise ValueError("The sampling rate should be 16000")
        if update_cfg is None:
            update_cfg = {"sr": sr}
        else:
            update_cfg["sr"] = sr
        self.prediction_mode = prediction_mode
        self.target_length = calc_target_length(sec=sec, sr=sr)
        model, _ = restore(ckpt_path=ckpt_path, update_cfg=update_cfg)
        return model

    def extract(self, batch: dict) -> PLOutput:
        x = batch["wave"]

        if self.device != x.device:
            logger.info("Move Model to the same device")
            self.device = x.device
            self.model.to(self.device)

        x = preprocess(source=x, target_length=self.target_length)
        feats = self.model.extract_features(
            x, mode="IMAGE", mask=False, remove_extra_tokens=False
        )
        if self.prediction_mode == "cls":
            z = feats["x"][:, 0]  # (B, L, D) -> (B, D)
        elif self.prediction_mode == "seq":
            z = feats["x"][:, 1:].mean(dim=1)  # (B, L, D) -> (B, D)
        else:
            raise ValueError(
                f"Unknown prediction mode {self.prediction_mode}, only cls and seq are supported"
            )
        embed_dict = {"main": z}
        return PLOutput(embed=embed_dict)
