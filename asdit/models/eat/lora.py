from dataclasses import dataclass

import torch
import torchaudio.compliance.kaldi as ta_kaldi

from fairseq.checkpoint_utils import load_model_ensemble_and_task
from fairseq.utils import import_user_module

from ..lora import BaseLoRA


@dataclass
class UserDirModule:
    user_dir: str


class EATLoRA(BaseLoRA):
    def __init__(
        self,
        ckpt_path: str = "pretrained_models/eat/EAT-base_epoch10_pt.pt",
        lora_cfg: dict = {"r": 64, "target_modules": ["qkv"]},
        embed_size: int = 128,
        last_layer: str = "linear",
        model_cfg: dict = {},
    ):
        super().__init__(
            ckpt_path=ckpt_path,
            lora_cfg=lora_cfg,
            embed_size=embed_size,
            last_layer=last_layer,
            model_cfg=model_cfg,
        )
        self.norm_mean = -4.268
        self.norm_std = 4.569

    def construct_model(self, ckpt_path: str):
        import_user_module(UserDirModule("/".join(__file__.split("/")[:-1] + ["EAT"])))
        model, cfg, task = load_model_ensemble_and_task([ckpt_path])
        return model[0], 768

    def preprocess(self, source: torch.Tensor) -> torch.Tensor:
        """
        Args:
            source (torch.Tensor): (B, L)

        Returns:
            fbank (torch.Tensor): (B, H, W)
        """
        fbanks = []
        source = source - source.mean(dim=-1, keepdim=True)
        for waveform in source:
            waveform = waveform.unsqueeze(0)
            fbank = ta_kaldi.fbank(
                waveform,
                htk_compat=True,
                sample_frequency=16000,
                use_energy=False,
                window_type="hanning",
                num_mel_bins=128,
                dither=0.0,
                frame_shift=10,
            )
            fbanks.append(fbank)
        fbank = torch.stack(fbanks, dim=0)
        return fbank

    def extract_features(self, x):
        """
        Args
            x: (B, L)
        """
        x = self.preprocess(x).unsqueeze(1)
        target_length = 1024
        diff = target_length - x.shape[-2]
        if diff > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, diff))
            x = m(x)
        elif diff < 0:
            x = x[:, 0:target_length, :]
        x = (x - self.norm_mean) / (self.norm_std * 2)
        feats = self.model.extract_features(
            x, padding_mask=None, mask=False, remove_extra_tokens=False
        )
        feats = feats["x"][:, :1]  # (B, 1, D)
        return feats

    # def _load_from_state_dict(
    #     self,
    #     state_dict: dict,
    #     prefix: str,
    #     local_metadata: dict,
    #     strict: bool,
    #     missing_keys: list,
    #     unexpected_keys: list,
    #     error_msgs: list,
    # ):
    #     # new_state_dict = state_dict
    #     super()._load_from_state_dict(
    #         state_dict,
    #         prefix="",
    #         local_metadata=local_metadata,
    #         strict=strict,
    #         missing_keys=missing_keys,
    #         unexpected_keys=unexpected_keys,
    #         error_msgs=error_msgs,
    #     )
