from typing import Literal, Optional

import torch

from asdkit.augmentations.specaug import spectrogram_augment

from ..wrapper import BaseLoRA
from .tools import calc_target_length, preprocess, restore


class EATLoRA(BaseLoRA):
    def __init__(
        self,
        lora_cfg: Optional[dict] = None,
        embed_size: int = 128,
        projection_type: str = "linear",
        model_cfg: Optional[dict] = None,
    ):
        if lora_cfg is None:
            lora_cfg = {"r": 64, "target_modules": ["qkv"]}
        super().__init__(
            lora_cfg=lora_cfg,
            embed_size=embed_size,
            projection_type=projection_type,
            model_cfg=model_cfg,
        )
        self.norm_mean = -4.268
        self.norm_std = 4.569

    def construct_model(
        self,
        sec: float,
        ckpt_path: str = "pretrained_models/eat/EAT-base_epoch10_pt.pt",
        sr: int = 16000,
        update_cfg: Optional[dict] = None,
        prediction_mode: Literal["cls", "seq"] = "cls",
        specaug: bool = False,
        specaug_time_prob: float = 0.5,
        specaug_time_width: int = 80,
        specaug_freq_prob: float = 0.5,
        specaug_freq_width: int = 80,
    ) -> tuple[torch.nn.Module, int]:
        if sr != 16000:
            raise ValueError("The sampling rate should be 16000")
        if update_cfg is None:
            update_cfg = {"sr": sr, "sec": sec}
        else:
            update_cfg["sr"] = sr
            update_cfg["sec"] = sec
        if self.projection_type == "attn_stat_pool" and prediction_mode == "cls":
            raise ValueError(
                "attn_stat_pool can be used with only <seq> prediction mode"
            )
        self.prediction_mode = prediction_mode
        self.target_length = calc_target_length(sec=sec, sr=sr)
        self.specaug = specaug
        if self.specaug:
            self.specaug_time_width = specaug_time_width
            self.specaug_time_prob = specaug_time_prob
            self.specaug_freq_width = specaug_freq_width
            self.specaug_freq_prob = specaug_freq_prob
        model, dim = restore(ckpt_path=ckpt_path, update_cfg=update_cfg)
        return model, dim

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args
            x: (B, L)

        Returns
            feats: (B, 1, D) or (B, L, D)
        """
        x = preprocess(
            source=x,
            target_length=self.target_length,
            norm_mean=self.norm_mean,
            norm_std=self.norm_std,
        )
        if self.training and self.specaug:
            x = spectrogram_augment(
                X=x,
                time_width=self.specaug_time_width,
                time_prob=self.specaug_time_prob,
                freq_width=self.specaug_freq_width,
                freq_prob=self.specaug_freq_prob,
                is_tf=True,
            )
        feats = self.model.extract_features(
            x, mode="IMAGE", mask=False, remove_extra_tokens=False
        )
        if self.prediction_mode == "cls":
            feats = feats["x"][:, :1]  # (B, 1, D)
        elif self.prediction_mode == "seq":
            feats = feats["x"][:, 1:]
        else:
            raise ValueError(
                f"Unknown prediction mode {self.prediction_mode}, only cls and seq are supported"
            )
        return feats
