from pathlib import Path
from typing import Optional, Union

import torch
from torch import nn

from asdkit.utils.common.optional_dependencies import require_optional
from asdkit.utils.common.representation import EmbeddingRepresentation


def transform_features(features: torch.Tensor, output_mode: str) -> torch.Tensor:
    """
    Args:
        features: Raw SSL features with shape (B, T, D).
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
        return features.unsqueeze(dim=2)  # [B, T, 1, D]
    raise ValueError(f"Unsupported Dasheng output_mode: {output_mode}")


class DashengSSLModel(nn.Module):
    """
    Args:
        model_id: Hugging Face model identifier or local pretrained directory.
        cache_dir: Hugging Face cache directory; None uses its standard settings.
        sr: Waveform sampling rate; must be 16000.

    Attributes:
        embedding_dim: Last dimension D of the raw SSL features.
        embedding_representation: Layout identifier used in extraction metadata.
    """

    embedding_representation: EmbeddingRepresentation = "dasheng"
    transform_features = staticmethod(transform_features)

    def __init__(
        self,
        model_id: str = "mispeech/dasheng-base",
        sr: int = 16000,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        if sr != 16000:
            raise ValueError("The sampling rate should be 16000")
        self.sr = sr
        transformers = require_optional(
            "transformers", "transformers", "dasheng", "Dasheng", ">=4.23"
        )
        self.feature_extractor = transformers.AutoFeatureExtractor.from_pretrained(
            model_id, cache_dir=cache_dir, trust_remote_code=True
        )
        model = transformers.AutoModel.from_pretrained(
            model_id, cache_dir=cache_dir, outputdim=None, trust_remote_code=True
        )
        super().__init__()
        self.embedding_dim = model.config.encoder_kwargs["embed_dim"]
        self.model = model

    def model_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the upstream encoder stages with corrected chunk handling."""
        # Upstream: https://huggingface.co/mispeech/dasheng-base/blob/main/modeling_dasheng.py#L368
        # Although standard DCASE2020-2026 clip lengths do not trigger this issue,
        # upstream drops the last chunk for longer inputs exactly divisible by target_length.
        # When multiple input clips are each split into chunks, upstream concatenation
        # mixes chunks from different clips; preserve each clip's chunk order instead.

        encoder = self.model.encoder
        x = x.unsqueeze(1)
        if encoder.init_bn is not None:
            x = encoder.init_bn(x)
        target = encoder.target_length
        stride = encoder.patch_stride[-1]
        if target % stride:
            raise ValueError(
                "Dasheng target_length must be divisible by its time patch stride"
            )
        if x.shape[-1] <= target:
            return encoder.forward_features(x)
        if encoder.pooling == "token":
            raise ValueError("Chunked Dasheng features require sequence pooling")
        length, batch_size = x.shape[-1], x.shape[0]
        chunks = x.split(target, dim=-1)
        chunks = [
            torch.nn.functional.pad(chunk, (0, target - chunk.shape[-1]))
            for chunk in chunks
        ]
        features = encoder.forward_features(torch.cat(chunks, dim=0))
        features = features.reshape(len(chunks), batch_size, -1, features.shape[-1])
        features = features.permute(1, 0, 2, 3).flatten(1, 2)
        return features[:, : length // stride]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Waveforms with shape (B, samples).

        Returns:
            Raw SSL features with shape (B, L, D).
        """
        inputs = self.feature_extractor(x, sampling_rate=self.sr, return_tensors="pt")
        return self.model_forward(inputs["input_values"])
