from typing import Dict, List, Optional

import torch
import torchaudio.transforms as T
from asdkit.utils.common import re_match_any
from torch import nn

from .utils import get_dec


def spectrogram_augment(
    X: torch.Tensor,
    time_width: int = 80,
    time_prob: float = 0.5,
    freq_width: int = 80,
    freq_prob: float = 0.5,
    iid: bool = True,
    is_tf=False,
) -> torch.Tensor:
    """
    Args:
        X (torch.Tensor): Input spectrogram (B, F, T)
        time_width (int): Width of time masking
        time_prob (float): Probability of applying time masking (0 <= time_prob <= 1)
        freq_width (int): Width of frequency masking
        freq_prob (float): Probability of applying frequency masking (0 <= freq_prob <= 1)
        iid (bool): Whether to use iid masks
        is_tf (bool): Whether the input is in (B, T, F) format (True) or (B, F, T) format (False)
    Returns:
        X (torch.Tensor): Augmented spectrogram with time and frequency masking applied (B, F, T)
    """
    if is_tf:
        X = X.transpose(-2, -1)  # (B, T, F) -> (B, F, T)

    time_masking = T.TimeMasking(time_mask_param=time_width, iid_masks=iid)
    dec = get_dec(len(X), time_prob, X.device)
    dec = dec.reshape([-1] + [1] * (len(X.shape) - 1))
    X_m = time_masking(X)
    X = dec * X_m + (1 - dec) * X

    freq_masking = T.FrequencyMasking(freq_mask_param=freq_width, iid_masks=iid)
    dec = get_dec(len(X), freq_prob, X.device)
    dec = dec.reshape([-1] + [1] * (len(X.shape) - 1))
    X_m = freq_masking(X)
    X = dec * X_m + (1 - dec) * X

    if is_tf:
        X = X.transpose(-2, -1)  # (B, F, T) -> (B, T, F)
    return X


class SpecAug(nn.Module):
    def __init__(
        self,
        time_width: int = 80,
        time_prob: float = 0.5,
        freq_width: int = 80,
        freq_prob: float = 0.5,
        iid: bool = True,
        stft_n_fft: int = 1024,
        stft_hop_length: Optional[int] = None,
        stft_window_type: str = "hann",
        target_keys: Optional[List[str]] = None,
    ):
        """
        Args:
            time_width (int): Width of time masking
            time_prob (float): Probability of applying time masking (0 <= time_prob <= 1)
            freq_width (int): Width of frequency masking
            freq_prob (float): Probability of applying frequency masking (0 <= freq_prob <= 1)
            iid (bool): Whether to use iid masks
            stft_n_fft (int): SpecAug applies STFT with this n_fft when input is waveform.
            stft_hop_length (int, optional): SpecAug applies STFT with this hop_length when input is waveform.
            stft_window_type (str): SpecAug applies STFT with this window type when input is waveform. ('hann' or 'hamming')
            target_keys (Optional[List[str]]): List of keys in the batch to apply SpecAugmentation. If None, defaults to ["wave"].
        """
        super().__init__()
        # Target keys
        if target_keys is None:
            target_keys = ["wave"]
        self.target_keys = target_keys

        # SpecAug parameters
        if not (0 <= time_prob <= 1):
            raise ValueError(f"prob should be in [0, 1], but got {time_prob}.")
        if not (0 <= freq_prob <= 1):
            raise ValueError(f"prob should be in [0, 1], but got {freq_prob}.")

        self.time_width = time_width
        self.time_prob = time_prob
        self.freq_width = freq_width
        self.freq_prob = freq_prob
        self.iid = iid

        # STFT parameters
        self.stft_n_fft = stft_n_fft
        self.stft_hop_length = stft_hop_length
        if stft_window_type == "hann":
            self.stft_window = torch.hann_window(self.stft_n_fft, True)
        elif stft_window_type == "hamming":
            self.stft_window = torch.hamming_window(self.stft_n_fft, True)
        else:
            raise ValueError("window_type must be either hann or hamming")

    def process(self, data: torch.Tensor) -> torch.Tensor:
        if len(data.shape) == 2:
            spectrograms_org = self.stft(data)
            spectrograms_aug = spectrogram_augment(
                X=torch.abs(spectrograms_org),
                time_width=self.time_width,
                time_prob=self.time_prob,
                freq_width=self.freq_width,
                freq_prob=self.freq_prob,
                iid=self.iid,
            )
            data_aug = self.istft(
                aug=spectrograms_aug, org=spectrograms_org, length=data.shape[-1]
            )
        elif len(data.shape) == 3:
            data_aug = spectrogram_augment(
                X=data,
                time_width=self.time_width,
                time_prob=self.time_prob,
                freq_width=self.freq_width,
                freq_prob=self.freq_prob,
                iid=self.iid,
            )
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}")
        return data_aug

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        if not self.training:
            return batch

        new_batch: Dict[str, torch.Tensor] = {}

        for key in batch:
            if re_match_any(patterns=self.target_keys, string=key):
                new_batch[key] = self.process(batch[key])
            else:
                new_batch[key] = batch[key]

        return new_batch

    def move_window_device(self, x: torch.Tensor):
        if self.stft_window.device != x.device:
            self.stft_window = self.stft_window.to(x.device)

    def stft(self, x):
        """
        Args:
            x (Tensor): input signal (B, T)
        Returns:
            X (Tensor): complex spectrogram (B, F, T)
        """
        self.move_window_device(x)
        X = torch.stft(
            x,
            n_fft=self.stft_n_fft,
            hop_length=self.stft_hop_length,
            win_length=self.stft_n_fft,
            window=self.stft_window,
            pad_mode="constant",
            onesided=True,
            return_complex=True,
        )
        return X

    def istft(self, aug, org, length):
        """
        Args:
            aug (Tensor): Augmented amplitude spectrogram (B, F, T)
            org (Tensor): Original complex spectrogram (B, F, T)
            length (int): length of the output signal
        Returns:
            x (Tensor): output signal (B, T)
        """
        X = aug * torch.exp(1j * torch.angle(org))
        self.move_window_device(X)
        x = torch.istft(
            X,
            n_fft=self.stft_n_fft,
            hop_length=self.stft_hop_length,
            win_length=self.stft_n_fft,
            window=self.stft_window,
            onesided=True,
            length=length,
        )
        return x
