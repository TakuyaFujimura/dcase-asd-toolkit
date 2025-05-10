from pathlib import Path

from ..lora import BaseLoRA, spectrogram_augment
from .tools import resume


class BEATsLoRA(BaseLoRA):
    def __init__(
        self,
        ckpt_path: str = "pretrained_models/beats/BEATs_iter3.pt",
        lora_cfg: dict = {"r": 64, "target_modules": ["q_proj", "v_proj"]},
        embed_size: int = 128,
        last_layer: str = "linear",
        model_cfg: dict = {},
        split10sec: bool = False,
    ):
        super().__init__(
            ckpt_path=ckpt_path,
            lora_cfg=lora_cfg,
            embed_size=embed_size,
            last_layer=last_layer,
            model_cfg=model_cfg,
            split10sec=split10sec,
        )

    def construct_model(
        self,
        ckpt_path: str,
        sr: int = 16000,
        update_cfg: dict = {},
        specaug: bool = False,
        specaug_freqm: int = 80,
        specaug_timem: int = 80,
    ):
        if sr != 16000:
            raise ValueError("The sampling rate should be 16000")
        self.specaug = specaug
        if self.specaug:
            self.specaug_freqm = specaug_freqm
            self.specaug_timem = specaug_timem
        model = resume(ckpt_path=Path(ckpt_path), update_cfg=update_cfg)
        return model, 768

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
