from typing import Optional

from torch import nn

from ..wrapper import BaseLoRA, spectrogram_augment
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
        specaug_freqm: int = 80,
        specaug_timem: int = 80,
    ) -> tuple[nn.Module, int]:
        if sr != 16000:
            raise ValueError("The sampling rate should be 16000")
        self.specaug = specaug
        if self.specaug:
            self.specaug_freqm = specaug_freqm
            self.specaug_timem = specaug_timem
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
                fbank,
                specaug_freqm=self.specaug_freqm,
                specaug_timem=self.specaug_timem,
            )
        return self.model.extract_features_from_fbank(fbank)[0]
