from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch import nn

from asdkit.constants import ASDKIT_CACHE_DIR
from asdkit.utils.common import instantiate
from asdkit.utils.common.representation import EmbeddingRepresentation

from .tools import restore


def transform_features(features: torch.Tensor, output_mode: str) -> torch.Tensor:
    """
    Args:
        features: Raw SSL features with shape (B, T * 8, D).
        output_mode: seq, mean, or time_frequency.

    Returns:
        Features with shape (B, tokens, D), (B, D), or (B, T, F, D).
    """
    assert features.ndim == 3
    if output_mode == "seq":
        return features
    if output_mode == "mean":
        return features.mean(dim=1)
    if output_mode == "time_frequency":
        # BEATs patch embedding has 8 frequency patches (128 mel bins / 16).
        batch_size, token_num, embedding_dim = features.shape
        assert token_num % 8 == 0
        return features.reshape(batch_size, -1, 8, embedding_dim)
    raise ValueError(f"Unsupported BEATs output_mode: {output_mode}")


class BEATsSSLModel(nn.Module):
    """
    Args:
        model_id: Checkpoint stem under cache_dir/beats.
        cache_dir: Weight directory, relative to the working directory by default.
        sr: Waveform sampling rate; must be 16000.
        update_cfg: Overrides for the pretrained BEATs configuration.
        spectrogram_augmentation_cfg: Optional Tensor augmentation configuration.
            Receives (B, T, F); use layout="TF" for TimeFrequencyMasking.
            None disables augmentation.

    Attributes:
        embedding_dim: Last dimension D of the raw SSL features.
        embedding_representation: Layout identifier used in extraction metadata.
    """

    embedding_representation: EmbeddingRepresentation = "beats"
    transform_features = staticmethod(transform_features)

    def __init__(
        self,
        model_id: str = "BEATs_iter3",
        sr: int = 16000,
        update_cfg: Optional[Dict[str, Any]] = None,
        spectrogram_augmentation_cfg: Optional[Dict[str, Any]] = None,
        cache_dir: Union[str, Path] = ASDKIT_CACHE_DIR,
    ) -> None:
        if sr != 16000:
            raise ValueError("The sampling rate should be 16000")
        ckpt_path = Path(cache_dir) / "beats" / f"{model_id}.pt"
        model, embedding_dim = restore(ckpt_path=ckpt_path, update_cfg=update_cfg)
        super().__init__()
        self.embedding_dim = embedding_dim
        self.model = model
        self.spectrogram_augmentation = instantiate(spectrogram_augmentation_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Waveforms with shape (B, samples).

        Returns:
            Raw SSL features with shape (B, L, D).
        """
        fbank = self.model.preprocess(x)
        if self.training and self.spectrogram_augmentation is not None:
            fbank = self.spectrogram_augmentation(fbank)
        features = self.model.extract_features_from_fbank(fbank)[0]
        return features
