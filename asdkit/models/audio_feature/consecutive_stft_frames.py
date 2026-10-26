from math import ceil
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .stft import Spectrogram


class ConsecutiveStftFrames(nn.Module):
    """Group frames with tail padding, or use unpadded windows with frame_shift."""

    def __init__(
        self, stft_cfg: Dict[str, Any], n_frames: int, frame_shift: Optional[int] = None
    ) -> None:
        """Args:
        stft_cfg: Spectrogram configuration.
        n_frames: Frames concatenated per vector.
        frame_shift: Window stride without padding; None groups frames
            with tail padding.
        """
        super().__init__()
        self.stft = Spectrogram(**stft_cfg)
        self.n_frames = n_frames
        if frame_shift is not None:
            assert frame_shift > 0
        self.frame_shift = frame_shift

    @property
    def dim(self) -> int:
        """Vector width after frequency cropping or Mel projection."""
        return self.stft.num_frequency_bins * self.n_frames

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: waveform [B, L]
        Returns:
            feature: n_frames consecutive spectrogram frames [B, n_vectors, n_frames*F]
        """
        X = self.stft(x)  # [B, F, T]
        B, F, T = X.shape

        if T < self.n_frames:
            raise ValueError(
                f"Input length T={T} is shorter than n_frames={self.n_frames}"
            )

        if self.frame_shift is not None:
            windows = X.unfold(-1, self.n_frames, self.frame_shift)
            # [B, F, T] -> [B, F, n_vectors, n_frames]
            feature = windows.permute(0, 2, 3, 1).reshape(B, -1, self.dim)
            # [B, n_vectors, n_frames, F] -> [B, n_vectors, n_frames*F]
            return feature

        n_vectors = ceil(T / self.n_frames)
        feature = X.new_zeros(B, F, n_vectors * self.n_frames)
        feature[:, :, :T] = X
        feature[:, :, -self.n_frames :] = X[:, :, -self.n_frames :]  # pad last frames
        feature = feature.reshape(B, F, n_vectors, self.n_frames).permute(0, 2, 3, 1)
        # [B, n_vectors, n_frames, F]
        feature = feature.reshape(B, n_vectors, self.n_frames * F)
        return feature
