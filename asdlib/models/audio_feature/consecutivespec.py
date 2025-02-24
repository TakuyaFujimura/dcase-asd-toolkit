from typing import Any, Dict

import torch

from .stft import STFT


class ConsecutiveSpec:
    def __init__(self, stft_cfg: Dict[str, Any], n_frames: int):
        self.stft = STFT(**stft_cfg)
        self.n_frames = n_frames
        self.output_freq_size = self.stft.output_freq_size

    def __call__(self, wave: torch.Tensor) -> torch.Tensor:
        """Make consecutive input feature
        Args:
            wave: [Batch, Time]
        Returns:
            x: [B*n_vectors, dim]
            where n_vectors is number of features obtained from a spectrogram,
            dim =  n_frames * n_fft , i.e., n_frames-consecutive spectrogram.
            content of x is like below,
            ---------------------------------
                    |n_fft |
                    -------------------------
            batch_0 |oooooo|oooooo|oooooo|
                    |...                 | n_vectors
                    |oooooo|oooooo|oooooo|
                    -------------------------
            batch_1 |oooooo|oooooo|oooooo|
                    |...                 |
                    |oooooo|oooooo|oooooo|
        """
        if len(wave.shape) != 2:
            raise ValueError("Input shape must be [Batch, Time]")

        batch_size = wave.shape[0]
        spectrograms = self.stft(wave)  # B, F, T
        n_vectors = spectrograms.shape[-1] - self.n_frames + 1
        n_fft = spectrograms.shape[1]
        if n_vectors <= 0:
            raise ValueError("n_frames is too large or length of wave is too short")
        x = torch.zeros((n_vectors * batch_size, n_fft * self.n_frames)).to(wave.device)
        for i in range(batch_size):
            for t in range(self.n_frames):
                x[
                    n_vectors * i : n_vectors * (i + 1),
                    n_fft * t : n_fft * (t + 1),
                ] = spectrograms[i, :, t : t + n_vectors].T
        return x
