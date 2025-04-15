import torch
import torchaudio.transforms as T
from torch import nn

from .utils import get_dec


class SpecAug(nn.Module):
    def __init__(
        self,
        time_width: int = 80,
        time_p: float = 0.5,
        freq_width: int = 80,
        freq_p: float = 0.5,
        iid: bool = True,
    ):
        super().__init__()

        if not (0 <= time_p <= 1):
            raise ValueError(f"time_p should be in [0, 1], but got {time_p}.")
        if not (0 <= freq_p <= 1):
            raise ValueError(f"freq_p should be in [0, 1], but got {freq_p}.")
        self.time_masking = T.TimeMasking(time_mask_param=time_width, iid_masks=iid)
        self.freq_masking = T.FrequencyMasking(
            freq_mask_param=freq_width, iid_masks=iid
        )
        self.time_p = time_p
        self.freq_p = freq_p

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X (torch.Tensor): spectrogram

        Returns:
            torch.Tensor: augmented spectrogram
        """
        dec = get_dec(len(X), self.time_p, X.device)
        dec = dec.reshape([-1] + [1] * (len(X.shape) - 1))
        X_m = self.time_masking(X)
        X = dec * X_m + (1 - dec) * X

        dec = get_dec(len(X), self.freq_p, X.device)
        dec = dec.reshape([-1] + [1] * (len(X.shape) - 1))
        X_m = self.freq_masking(X)
        X = dec * X_m + (1 - dec) * X
        return X
