from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch import nn

from asdkit.utils.common import instantiate
from asdkit.utils.common.representation import EmbeddingRepresentation

from .tools import preprocess, restore


def transform_features(features: torch.Tensor, output_mode: str) -> torch.Tensor:
    """
    Args:
        features: Raw SSL features with shape (B, 1 + T * 8, D), including the leading CLS token.
        output_mode: cls, seq, mean, or time_frequency.

    Returns:
        Features with shape (B, tokens, D), (B, D), or (B, T, F, D).
    """
    assert features.ndim == 3
    if output_mode == "cls":
        return features[:, 0]
    sequence = features[:, 1:]
    if output_mode == "seq":
        return sequence
    if output_mode == "mean":
        return sequence.mean(dim=1)
    if output_mode == "time_frequency":
        # Remove the EAT CLS token before recovering the 8 frequency patches.
        batch_size, token_num, embedding_dim = sequence.shape
        assert token_num % 8 == 0
        return sequence.reshape(batch_size, -1, 8, embedding_dim)
    raise ValueError(f"Unsupported EAT output_mode: {output_mode}")


class EATSSLModel(nn.Module):
    """
    Args:
        model_id: Hugging Face model identifier or local pretrained directory.
        cache_dir: Hugging Face cache directory; None uses its standard settings.
        sr: Waveform sampling rate; must be 16000.
        pad_to_multiple: Zero-pad fbank time frames before normalization.
            None disables padding.
        spectrogram_augmentation_cfg: Optional Tensor augmentation configuration.
            Receives (B, 1, T, F); use layout="TF" for TimeFrequencyMasking.
            None disables augmentation.

    Attributes:
        embedding_dim: Last dimension D of the raw SSL features.
        embedding_representation: Layout identifier used in extraction metadata.
    """

    embedding_representation: EmbeddingRepresentation = "eat"
    transform_features = staticmethod(transform_features)

    def __init__(
        self,
        model_id: str = "worstchan/EAT-base_epoch30_pretrain",
        sr: int = 16000,
        spectrogram_augmentation_cfg: Optional[Dict[str, Any]] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        # Experimentally, padding improved EAT performance on DCASE 2023-2025,
        # while BEATs and Dasheng showed no consistent gain; 32-frame multiples
        # performed better overall than 16. For 10 s audio, 998 fbank frames
        # become 1024, matching the official EAT recommendation. Other lengths
        # also align with patch size 16, so no trailing fbank frames are discarded.
        pad_to_multiple: Optional[int] = 32,
    ) -> None:
        if sr != 16000:
            raise ValueError("The sampling rate should be 16000")
        model, embedding_dim = restore(model_id=model_id, cache_dir=cache_dir)
        super().__init__()
        self.embedding_dim = embedding_dim
        self.model = model
        self.spectrogram_augmentation = instantiate(spectrogram_augmentation_cfg)
        self.norm_mean = -4.268
        self.norm_std = 4.569
        self.pad_to_multiple = pad_to_multiple

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 16 kHz waveforms with shape (B, samples).

        Returns:
            Raw SSL features with shape (B, L, D)
        """
        x = preprocess(
            source=x,
            norm_mean=self.norm_mean,
            norm_std=self.norm_std,
            pad_to_multiple=self.pad_to_multiple,
        )
        if self.training and self.spectrogram_augmentation is not None:
            x = self.spectrogram_augmentation(x)
        features = self.model.extract_features(x)
        return features
