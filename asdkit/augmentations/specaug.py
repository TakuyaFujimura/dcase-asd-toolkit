from typing import Dict, List, Literal, Optional

import torch
import torchaudio.transforms as T
from torch import nn

from asdkit.utils.common import re_match_any

from .base import BaseBatchAugmentation
from .utils import get_dec


class TimeFrequencyMasking(nn.Module):
    """Mask spectrograms shaped (..., F, T), or (..., T, F) with layout="TF".

    Inputs include a leading batch dimension, e.g. (B, F, T) or (B, 1, F, T).
    Each masking probability controls whether that mask is applied per sample.
    Evaluation returns the input unchanged.
    """

    def __init__(
        self,
        time_width: int = 80,
        time_prob: float = 0.5,
        freq_width: int = 80,
        freq_prob: float = 0.5,
        iid: bool = True,
        layout: Literal["FT", "TF"] = "FT",
    ) -> None:
        """Initialize time and frequency masking for spectrogram tensors.

        Args:
            time_width: Upper bound on time mask width, measured in frames.
            time_prob: Probability of applying time masking to each sample,
                in [0, 1].
            freq_width: Upper bound on frequency mask width, measured in bins.
            freq_prob: Probability of applying frequency masking to each sample,
                in [0, 1].
            iid: Whether torchaudio samples masks independently for each
                example/channel rather than sharing a mask across them.
            layout: Order of the last two input dimensions: "FT" for
                (..., F, T), or "TF" for (..., T, F).
        """
        super().__init__()
        if not (0 <= time_prob <= 1):
            raise ValueError(f"prob should be in [0, 1], but got {time_prob}.")
        if not (0 <= freq_prob <= 1):
            raise ValueError(f"prob should be in [0, 1], but got {freq_prob}.")
        self.time_prob = time_prob
        self.freq_prob = freq_prob
        assert layout in ("FT", "TF")
        self.layout = layout
        self.time_masking = T.TimeMasking(time_mask_param=time_width, iid_masks=iid)
        self.freq_masking = T.FrequencyMasking(
            freq_mask_param=freq_width, iid_masks=iid
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        if self.layout == "TF":
            x = x.transpose(-2, -1)  # (B, T, F) -> (B, F, T)

        dec = get_dec(batch_size=len(x), prob=self.time_prob, device=x.device)
        dec = dec.reshape([-1] + [1] * (x.ndim - 1))
        masked = self.time_masking(x)
        x = dec * masked + (1 - dec) * x

        dec = get_dec(batch_size=len(x), prob=self.freq_prob, device=x.device)
        dec = dec.reshape([-1] + [1] * (x.ndim - 1))
        masked = self.freq_masking(x)
        x = dec * masked + (1 - dec) * x

        if self.layout == "TF":
            x = x.transpose(-2, -1)  # (B, F, T) -> (B, T, F)
        return x


class SpecAugment(BaseBatchAugmentation):
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
    ) -> None:
        """
        This class applies SpecAugment to the input data.
        If the input is a waveform, STFT and iSTFT are used for the transformation.
        For SSL models, the SpecAugment parameters should be specified in the model configuration.

        Args:
            time_width (int): Width of time masking
            time_prob (float): Probability of applying time masking (0 <= time_prob <= 1)
            freq_width (int): Width of frequency masking
            freq_prob (float): Probability of applying frequency masking (0 <= freq_prob <= 1)
            iid (bool): Whether to use iid masks
            stft_n_fft (int): SpecAug applies STFT with this n_fft when input is waveform.
            stft_hop_length (int, optional): SpecAug applies STFT with this hop_length when input is waveform.
            stft_window_type (str): SpecAug applies STFT with this window type when input is waveform. ('hann' or 'hamming')
            target_keys (Optional[List[str]]): List of keys in the batch to which SpecAug will be applied. If None, defaults to ["wave"].
        """
        super().__init__()
        # Target keys
        if target_keys is None:
            target_keys = ["wave"]
        self.target_keys = target_keys

        # SpecAug parameters
        self.masking = TimeFrequencyMasking(
            time_width=time_width,
            time_prob=time_prob,
            freq_width=freq_width,
            freq_prob=freq_prob,
            iid=iid,
        )

        # STFT parameters
        self.stft_n_fft = stft_n_fft
        self.stft_hop_length = stft_hop_length
        if stft_window_type == "hann":
            window = torch.hann_window(self.stft_n_fft, True)
        elif stft_window_type == "hamming":
            window = torch.hamming_window(self.stft_n_fft, True)
        else:
            raise ValueError("window_type must be either hann or hamming")
        self.register_buffer("stft_window", window, persistent=False)

    def process(self, data: torch.Tensor) -> torch.Tensor:
        if len(data.shape) == 2:
            spectrograms_org = self.stft(data)
            spectrograms_aug = self.masking(torch.abs(spectrograms_org))
            data_aug = self.istft(
                aug=spectrograms_aug, org=spectrograms_org, length=data.shape[-1]
            )
        elif len(data.shape) > 2:
            data_aug = self.masking(data)
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

    def stft(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): input signal (B, T)
        Returns:
            X (Tensor): complex spectrogram (B, F, T)
        """
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

    def istft(self, aug: torch.Tensor, org: torch.Tensor, length: int) -> torch.Tensor:
        """
        Args:
            aug (Tensor): Augmented amplitude spectrogram (B, F, T)
            org (Tensor): Original complex spectrogram (B, F, T)
            length (int): length of the output signal
        Returns:
            x (Tensor): output signal (B, T)
        """
        X = aug * torch.exp(1j * torch.angle(org))
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
