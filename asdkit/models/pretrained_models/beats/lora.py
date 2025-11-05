from typing import Optional

from asdkit.augmentations.specaug import spectrogram_augment
from torch import nn

from ..wrapper import BaseLoRA
from .tools import restore


class BEATsLoRA(BaseLoRA):
    def __init__(
        self,
        lora_cfg: Optional[dict] = None,
        embed_size: int = 128,
        projection_type: str = "linear",
        model_cfg: Optional[dict] = None,
    ):
        if lora_cfg is None:
            lora_cfg = {"r": 64, "target_modules": ["q_proj", "v_proj"]}
        super().__init__(
            lora_cfg=lora_cfg,
            embed_size=embed_size,
            projection_type=projection_type,
            model_cfg=model_cfg,
        )

    def construct_model(
        self,
        ckpt_path: str = "pretrained_models/beats/BEATs_iter3.pt",
        sr: int = 16000,
        update_cfg: Optional[dict] = None,
        specaug: bool = False,
        specaug_time_prob: float = 0.5,
        specaug_time_width: int = 80,
        specaug_freq_prob: float = 0.5,
        specaug_freq_width: int = 80,
    ) -> tuple[nn.Module, int]:
        if sr != 16000:
            raise ValueError("The sampling rate should be 16000")
        self.specaug = specaug
        if self.specaug:
            self.specaug_time_width = specaug_time_width
            self.specaug_time_prob = specaug_time_prob
            self.specaug_freq_width = specaug_freq_width
            self.specaug_freq_prob = specaug_freq_prob
        model, dim = restore(ckpt_path=ckpt_path, update_cfg=update_cfg)
        return model, dim

    def extract_features(self, x):
        """
        Args
            x: (B, L)
        """
        fbank = self.model.preprocess(x)
        if self.training and self.specaug:
            fbank = spectrogram_augment(
                X=fbank,
                time_width=self.specaug_time_width,
                time_prob=self.specaug_time_prob,
                freq_width=self.specaug_freq_width,
                freq_prob=self.specaug_freq_prob,
                is_tf=True,
            )
        return self.model.extract_features_from_fbank(fbank)[0]
