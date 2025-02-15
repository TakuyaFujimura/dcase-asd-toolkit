from typing import List

import torch
from torch import nn

from .fft_encoder import FFTEncoderLayer
from .stft_encoder import STFTEncoderLayer


class MultiResNet(nn.Module):
    def __init__(
        self,
        sec: int,
        sr: int,
        stft_cfg_list: List[dict],
        use_bias: bool = False,
        emb_base_size: int = 128,
        resnet_additional_layer: str = "normal",
    ):
        super().__init__()
        self.fft_layer = FFTEncoderLayer(
            sec=sec, sr=sr, use_bias=use_bias, emb_base_size=emb_base_size
        )
        self.stft_layer_list = nn.ModuleList([])
        for stft_cfg in stft_cfg_list:
            stft_encoder = STFTEncoderLayer(
                sec=sec,
                sr=sr,
                stft_cfg=stft_cfg,
                use_bias=use_bias,
                emb_base_size=emb_base_size,
                resnet_additional_layer=resnet_additional_layer,
            )
            self.stft_layer_list.append(stft_encoder)

    def forward(self, x_time: torch.Tensor):
        """
        Args
            x_time: (B, L)

        Returns
            z: (B, emb_base_size * (1 + len(stft_cfg_list)))
        """
        z_list: List[torch.Tensor] = [self.fft_layer(x_time)]
        for stft_layer in self.stft_layer_list:
            z_list += [stft_layer(x_time)]
        return torch.cat(z_list, dim=-1)
