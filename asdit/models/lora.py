from abc import ABC, abstractmethod

import torch
import torchaudio
from torch import nn

from .modules import AttnStatPool, add_lora


def spectrogram_augment(
    spec: torch.Tensor, specaug_freqm: int = 80, specaug_timem: int = 80
) -> torch.Tensor:
    freq_masking = torchaudio.transforms.FrequencyMasking(specaug_freqm, iid_masks=True)
    time_masking = torchaudio.transforms.TimeMasking(specaug_timem, iid_masks=True)
    spec_ = spec.transpose(-2, -1)
    input_with_freq_mask = freq_masking(spec_)
    input_with_time_freq_mask = time_masking(input_with_freq_mask)
    input_with_time_freq_mask = torch.transpose(input_with_time_freq_mask, -2, -1)
    return input_with_time_freq_mask


class BaseLoRA(nn.Module, ABC):
    def __init__(
        self,
        ckpt_path: str,
        lora_cfg: dict,
        embed_size: int = 128,
        last_layer: str = "linear",
        model_cfg: dict = {},
        split10sec: bool = False,
    ):
        super().__init__()
        self.split10sec = split10sec
        model, feature_dim = self.construct_model(ckpt_path, **model_cfg)
        self.model = add_lora(model, lora_cfg)
        self.embed_size = embed_size
        self.last_layer = last_layer
        if self.last_layer == "linear":
            self.network = nn.Linear(feature_dim, self.embed_size)
        elif self.last_layer == "attn_stat_pool":
            self.network = nn.Sequential(
                AttnStatPool(embed_size=feature_dim),
                nn.Linear(feature_dim, self.embed_size),
            )
        else:
            raise NotImplementedError(f"last_layer={self.last_layer} is not supported.")

    @abstractmethod
    def construct_model(self, ckpt_path: str, **kwargs) -> tuple[nn.Module, int]:
        """
        Args:
            ckpt_path (str): path to the checkpoint file

        Returns:
            SSL Model[nn.Module]: ssl model
            feature dim [int]: feature dimension of the model
        """
        pass

    @abstractmethod
    def extract_features(self, x):
        pass

    def forward(self, x):
        """
        Args
            x: (B, L)
        """
        if self.split10sec:
            size = 16000 * 10
            x_list = [x[i * size : (i + 1) * size] for i in range(x.shape[-1] // size)]
            if x.shape[-1] % size != 0:
                x_list.append(x[-size:])
        else:
            x_list = [x]

        z_list = []
        for x in x_list:
            z = self.extract_features(x)
            z_list.append(z)
        z = torch.stack(z_list, dim=0).mean(dim=0)

        if self.last_layer == "linear":
            z = self.network(z)  # (B, L, D)
            z = torch.mean(z, dim=1)  # (B, D)
        elif self.last_layer == "attn_stat_pool":
            z = self.network(z)  # (B, D)
        else:
            raise NotImplementedError(f"last_layer={self.last_layer} is not supported.")

        return z
