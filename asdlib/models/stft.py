import sys

import torch
import torchaudio.transforms as T
from torch import nn


class STFT(nn.Module):
    def __init__(
        self,
        use_mel: bool,
        sr: int,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        power: float,
        f_min: float,
        f_max: float,
        use_log: bool = False,
        temporal_norm: bool = False,
    ):
        super().__init__()
        self.use_mel = use_mel
        self.n_fft = n_fft
        self.sr = sr
        self.f_min = f_min
        self.f_max = f_max
        self.power = power
        self.use_log = use_log
        self.temporal_norm = temporal_norm
        if use_mel:
            self.stft = T.MelSpectrogram(
                sample_rate=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                f_min=f_min,
                f_max=f_max,
                pad=0,
                n_mels=n_mels,
                power=power,
                normalized=True,
                center=True,
                pad_mode="reflect",
                onesided=True,
            )
        else:
            self.stft = T.Spectrogram(
                n_fft=n_fft,
                hop_length=hop_length,
                pad=0,
                power=power,
                normalized=True,
                center=True,
                pad_mode="reflect",
                onesided=True,
            )

    def forward(self, x):
        # B, F, T
        spectrogram = self.stft(x)
        if not self.use_mel:
            frequencies = torch.linspace(0, self.sr // 2, self.n_fft // 2 + 1)
            if self.f_min is not None:
                f_min_idx = torch.searchsorted(frequencies, self.f_min, right=False)
            else:
                f_min_idx = 0
            if self.f_max is not None:
                f_max_idx = torch.searchsorted(frequencies, self.f_max, right=True)
            else:
                f_max_idx = None
            spectrogram = spectrogram[..., f_min_idx:f_max_idx, :]

        if self.use_log:
            spectrogram = (
                20.0
                / self.power
                * torch.log10(
                    torch.maximum(
                        spectrogram,
                        torch.tensor([sys.float_info.epsilon]).to(spectrogram.device),
                    )
                )
            )

        if self.temporal_norm:
            spectrogram -= torch.mean(spectrogram, dim=-1, keepdim=True)

        return spectrogram


class FFT(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Args:
            x: (B, T)
        """
        x_fft = torch.fft.rfft(x)
        x_abs = torch.abs(x_fft)
        return x_abs
