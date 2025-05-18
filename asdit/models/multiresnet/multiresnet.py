from typing import List

import torch
from torch import nn

from asdit.models.audio_feature import FFT
from asdit.models.audio_feature.stft import STFT
from asdit.utils.common import instantiate_tgt


class STFTEncoderLayer(nn.Module):
    def __init__(
        self,
        sec: float,
        sr: int,
        stft_cfg: dict,
        network_cfg: dict,
    ):
        super().__init__()
        self.stft = STFT(**stft_cfg)
        spectrogram_size = self.stft(torch.randn(int(sec * sr))).shape
        self.layer = instantiate_tgt({**network_cfg, "input_size": spectrogram_size})

    def forward(self, x):
        """
        Args
            x: wave (B, T)
        """
        x = self.stft(x)
        return self.layer(x)


class FFTEncoderLayer(nn.Module):
    def __init__(self, sec: float, sr: int, network_cfg: dict):
        super().__init__()
        fft_len = int(sec * sr) // 2 + 1
        self.layer = instantiate_tgt({**network_cfg, "input_size": fft_len})
        self.fft = FFT()

    def forward(self, x):
        """
        x: wave (B, L)
        """
        x = self.fft(x)
        x = x.unsqueeze(1)
        return self.layer(x)


class MultiResNet(nn.Module):
    def __init__(
        self,
        sec: float,
        sr: int,
        use_fft: bool,
        stft_cfg_list: List[dict],
        fft_network_cfg: dict,
        stft_network_cfg: dict,
        use_bias: bool = False,
        emb_base_size: int = 128,
    ):
        super().__init__()
        fft_network_cfg["use_bias"] = use_bias
        fft_network_cfg["emb_base_size"] = emb_base_size
        stft_network_cfg["use_bias"] = use_bias
        stft_network_cfg["emb_base_size"] = emb_base_size

        if use_fft:
            self.fft_layer = FFTEncoderLayer(
                sec=sec, sr=sr, network_cfg=fft_network_cfg
            )
        else:
            self.fft_layer = None

        self.stft_layer_list = nn.ModuleList([])
        for stft_cfg in stft_cfg_list:
            stft_encoder = STFTEncoderLayer(
                sec=sec,
                sr=sr,
                stft_cfg=stft_cfg,
                network_cfg=stft_network_cfg,
            )
            self.stft_layer_list.append(stft_encoder)
        if use_fft:
            self.embed_size = emb_base_size * (1 + len(stft_cfg_list))
        else:
            self.embed_size = emb_base_size * len(stft_cfg_list)

    def forward(self, x_time: torch.Tensor):
        """
        Args
            x_time: (B, L)

        Returns
            z: (B, emb_base_size * (1 + len(stft_cfg_list)))
        """
        z_list: List[torch.Tensor] = []
        if self.fft_layer is not None:
            z_list += [self.fft_layer(x_time)]
        for stft_layer in self.stft_layer_list:
            z_list += [stft_layer(x_time)]
        return torch.cat(z_list, dim=-1)
