import sys
from typing import Optional

import torch
import torchaudio.transforms as T
from torch import nn


class Spectrogram(nn.Module):
    """Extract one-sided magnitude/power spectrograms, optionally on a Mel scale.

    Uses a periodic Hann window, window-energy normalization, and centered
    frames. Linear spectra are cropped to the inclusive f_min/f_max range;
    Mel spectra use those bounds to construct the filterbank. Optional log
    scaling computes 20 / power * log10(spectrum), followed by optional
    subtraction of each frequency bin's mean over time.
    """

    def __init__(
        self,
        sr: int,
        n_fft: int,
        hop_length: int,
        power: float,
        f_min: Optional[float],
        f_max: Optional[float],
        n_mels: Optional[int] = None,
        use_log: bool = False,
        temporal_norm: bool = False,
        pad_mode: str = "constant",
    ) -> None:
        """
        Args:
            sr: Sample rate in hertz.
            n_fft: FFT size and Hann window length in samples.
            hop_length: Frame spacing in samples.
            power: Magnitude exponent (1 for amplitude, 2 for power).
            f_min: Lower frequency bound in hertz; None leaves linear spectra unbounded.
            f_max: Upper frequency bound in hertz; None uses the Nyquist limit.
            n_mels: Mel bin count, or None for a linear-frequency spectrum.
            use_log: Apply log scaling with a floor of sys.float_info.epsilon.
            temporal_norm: Subtract each bin's mean over time.
            pad_mode: Padding mode for centered frames.
        """
        super().__init__()

        self.n_fft = n_fft
        self.n_mels = n_mels
        self.use_mel = n_mels is not None
        self.sr = sr
        self.f_min = f_min
        self.f_max = f_max
        self.power = power
        self.use_log = use_log
        self.temporal_norm = temporal_norm
        if self.use_mel:
            assert self.n_mels is not None
            self.num_frequency_bins = self.n_mels
            self.stft = T.MelSpectrogram(
                sample_rate=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                f_min=f_min,
                f_max=f_max,
                pad=0,
                n_mels=self.n_mels,
                power=power,
                normalized=True,
                center=True,
                pad_mode=pad_mode,
            )
        else:
            frequencies = torch.linspace(0, sr // 2, n_fft // 2 + 1)
            self.f_min_idx = (
                int(torch.searchsorted(frequencies, f_min, right=False))
                if f_min is not None
                else 0
            )
            self.f_max_idx = (
                int(torch.searchsorted(frequencies, f_max, right=True))
                if f_max is not None
                else len(frequencies)
            )
            self.num_frequency_bins = max(0, self.f_max_idx - self.f_min_idx)
            self.stft = T.Spectrogram(
                n_fft=n_fft,
                hop_length=hop_length,
                pad=0,
                power=power,
                normalized=True,
                center=True,
                pad_mode=pad_mode,
                onesided=True,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map real waveforms (..., samples) to real features (..., F, frames).

        F equals num_frequency_bins after cropping or Mel projection. Log
        scaling promotes float16/bfloat16 features to float32; other dtypes
        are preserved. Move this module and the input to the same device.
        """
        spectrogram = self.stft(x)  # B, F, T
        if not self.use_mel:
            spectrogram = spectrogram[..., self.f_min_idx : self.f_max_idx, :]

        if self.use_log:
            # Compute the logarithm in at least float32 to keep the floor representable.
            if spectrogram.dtype in (torch.float16, torch.bfloat16):
                spectrogram = spectrogram.float()
            spectrogram = (
                20.0
                / self.power
                * torch.log10(spectrogram.clamp_min(sys.float_info.epsilon))
            )

        if self.temporal_norm:
            spectrogram -= torch.mean(spectrogram, dim=-1, keepdim=True)

        return spectrogram


class FFTMagnitude(nn.Module):
    """Extract unnormalized one-sided FFT magnitudes without windowing."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Real waveforms shaped (..., samples).

        Returns:
            Magnitudes shaped (..., samples // 2 + 1). Unlike Spectrogram, this
            transform applies no window, frequency crop, or log scaling.
        """
        x_fft = torch.fft.rfft(x)
        x_abs = torch.abs(x_fft)
        return x_abs
