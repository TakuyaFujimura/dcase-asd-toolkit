"""Construct fixed-width FFT/STFT branches and concatenate their embeddings."""

from typing import Any, Dict, List

import torch
from torch import nn

from asdkit.models.audio_feature import FFTMagnitude, Spectrogram
from asdkit.utils.common import instantiate


class STFTEncoderLayer(nn.Module):
    def __init__(
        self,
        sec: float,
        sr: int,
        stft_cfg: Dict[str, Any],
        network_cfg: Dict[str, Any],
        use_bias: bool,
        emb_base_size: int,
    ) -> None:
        """
        Args:
            sec: Waveform duration in seconds.
            sr: Sampling rate in hertz.
            stft_cfg: Spectrogram extractor configuration.
            network_cfg: Encoder constructor accepting input_size=(F, T).
                Additional architecture options come from this configuration.
                Forward maps (B, F, T) to (B, emb_base_size). Geometry is measured
                from the actual transform output, including frequency cropping.
            use_bias: Conv/Linear bias and BatchNorm affine policy; overrides config.
            emb_base_size: Internal and output width; overrides config.
        """
        super().__init__()
        self.stft = Spectrogram(**stft_cfg)
        spectrogram_size = self.stft(torch.zeros(int(sec * sr))).shape
        self.layer = instantiate(
            cfg=network_cfg,
            input_size=spectrogram_size,
            use_bias=use_bias,
            emb_base_size=emb_base_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Waveforms shaped ``[batch, samples]``.

        Returns:
            Embeddings shaped ``[batch, branch_embedding_dim]``.
        """
        x = self.stft(x)
        return self.layer(x)


class FFTEncoderLayer(nn.Module):
    def __init__(
        self,
        sec: float,
        sr: int,
        network_cfg: Dict[str, Any],
        use_bias: bool,
        emb_base_size: int,
    ) -> None:
        """
        Args:
            sec: Waveform duration in seconds.
            sr: Sampling rate in hertz.
            network_cfg: Encoder constructor accepting integer input_size (FFT bins).
                Additional architecture options come from this configuration.
                Forward maps (B, 1, input_size) to (B, emb_base_size).
            use_bias: Conv/Linear bias and BatchNorm affine policy; overrides config.
            emb_base_size: Internal and output width; overrides config.
        """
        super().__init__()
        fft_len = int(sec * sr) // 2 + 1
        self.layer = instantiate(
            cfg=network_cfg,
            input_size=fft_len,
            use_bias=use_bias,
            emb_base_size=emb_base_size,
        )
        self.fft = FFTMagnitude()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Waveforms shaped ``[batch, samples]``.

        Returns:
            Embeddings shaped ``[batch, branch_embedding_dim]``.
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
        stft_cfg_list: List[Dict[str, Any]],
        fft_network_cfg: Dict[str, Any],
        stft_network_cfg: Dict[str, Any],
        use_bias: bool = False,
        emb_base_size: int = 128,
    ) -> None:
        """
        T. Fujimura et al., "Improvements of discriminative feature space training for anomalous sound detection in unlabeled conditions," Proc. ICASSP, 2025.

        Args:
            sec: Waveform duration in seconds.
            sr: Sampling rate in hertz.
            use_fft: Include the FFT branch.
            stft_cfg_list: Spectrogram extractor configurations, one per branch.
            fft_network_cfg: FFT encoder target accepting input_size (bin count),
                use_bias and emb_base_size; maps (B, 1, bins) to (B, emb_base_size).
            stft_network_cfg: STFT encoder target accepting input_size=(F, T),
                use_bias and emb_base_size; maps (B, F, T) to (B, emb_base_size).
                A separate encoder is constructed for each STFT configuration.
            use_bias: Shared Conv/Linear bias and BatchNorm affine policy.
                Overrides the same key in both network configs. SEBlock retains
                its own internal bias policy.
            emb_base_size: Shared branch output and internal network width.
                Overrides the same key in both network configs. Kept coupled
                to preserve existing configurations and checkpoint shapes.
        """
        super().__init__()
        # Shared architecture options deliberately take precedence over child configs.
        if use_fft:
            self.fft_layer = FFTEncoderLayer(
                sec=sec,
                sr=sr,
                network_cfg=fft_network_cfg,
                use_bias=use_bias,
                emb_base_size=emb_base_size,
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
                use_bias=use_bias,
                emb_base_size=emb_base_size,
            )
            self.stft_layer_list.append(stft_encoder)
        self.embedding_dim = emb_base_size * (int(use_fft) + len(stft_cfg_list))

    def forward(self, x_time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_time: Waveforms shaped ``[batch, samples]``.

        Returns:
            Embeddings shaped ``[batch, embedding_dim]``; MultiResNet concatenates
            branch outputs.
        """
        z_list: List[torch.Tensor] = []
        if self.fft_layer is not None:
            z_list.append(self.fft_layer(x_time))
        for stft_layer in self.stft_layer_list:
            z_list.append(stft_layer(x_time))
        return torch.cat(z_list, dim=-1)
