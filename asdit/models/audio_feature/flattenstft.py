import logging
from typing import Any, Dict

import torch

from .base import BaseAudioFeature
from .stft import STFT

logger = logging.getLogger(__name__)


class FlattenSTFT(BaseAudioFeature):
    def __init__(self, stft_cfg: Dict[str, Any], n_frames: int):
        self.stft = STFT(**stft_cfg)
        self.n_frames = n_frames
        self.output_freq_size = self.stft.output_freq_size
        self.feat_dim = self.output_freq_size * self.n_frames

    def __call__(self, wave: torch.Tensor) -> torch.Tensor:
        """Make consecutive input feature
        Args:
            wave: [Batch, Time]
        Returns:
            x: [B*n_vectors, self.feat_dim]
            where n_vectors is number of features obtained from a spectrogram,
            self.feat_dim =  n_frames * n_fft , i.e., n_frames-consecutive spectrogram.

            x is like below (self.n_frames = 3):
            ----------------------------------------------------------------------------
                    |n_fft     |
                    ----------------------------------
            batch_0 |frame[0]  |frame[1]  |frame[2]  |
                    |frame[1]  |frame[2]  |frame[3]  |
                    |...                             | n_vectors = T - self.n_frames + 1
                    |frame[T-4]|frame[T-3]|frame[T-2]|
                    |frame[T-3]|frame[T-2]|frame[T-1]| (len(frame) = T, frame[T-1] is the last frame)
                    ----------------------------------
            batch_1 |frame[0]  |frame[1]  |frame[2]  |
                    |frame[1]  |frame[2]  |frame[3]  |
                    |...                             |
                    |frame[T-4]|frame[T-3]|frame[T-2]|
                    |frame[T-3]|frame[T-2]|frame[T-1]|
                    ----------------------------------
                    ...
        """
        if len(wave.shape) != 2:
            logger.error("Input shape must be [Batch, Time]")
            raise ValueError("Input shape must be [Batch, Time]")

        batch_size = wave.shape[0]
        spectrograms = self.stft(wave)  # B, F, T
        n_vectors = spectrograms.shape[-1] - self.n_frames + 1
        n_fft = spectrograms.shape[1]
        if n_vectors <= 0:
            logger.error("n_frames is too large or length of wave is too short")
            raise ValueError("n_frames is too large or length of wave is too short")
        x = torch.zeros((n_vectors * batch_size, n_fft * self.n_frames)).to(wave.device)
        for i in range(batch_size):
            for t in range(self.n_frames):
                x[
                    n_vectors * i : n_vectors * (i + 1),
                    n_fft * t : n_fft * (t + 1),
                ] = spectrograms[i, :, t : t + n_vectors].T
        return x
