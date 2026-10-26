from typing import Any, Callable, Dict

import torch
from torch import nn

from asdkit.utils.common import instantiate
from asdkit.utils.common.representation import EmbeddingRepresentation

from .adaptation import Frozen, PretrainedAdaptation


class PretrainedWrapper(nn.Module):
    """
    Args:
        model_cfg: SSL model configuration with _target_.
        adaptation_cfg: Frozen, LoRA, or full fine-tuning configuration.

    Attributes:
        embedding_dim: Last dimension of the raw SSL features.
        embedding_representation: Layout identifier for extraction metadata.
        transform_features: Convert raw features using features and output_mode.
    """

    def __init__(
        self, model_cfg: Dict[str, Any], adaptation_cfg: Dict[str, Any]
    ) -> None:
        super().__init__()
        model = instantiate(model_cfg)
        adaptation = instantiate(adaptation_cfg)
        if not isinstance(adaptation, PretrainedAdaptation):
            raise TypeError("adaptation_cfg must instantiate PretrainedAdaptation")
        self.embedding_dim = model.embedding_dim  # type: ignore[attr-defined]
        self.embedding_representation: EmbeddingRepresentation = (
            model.embedding_representation
        )
        self.transform_features: Callable[..., torch.Tensor] = model.transform_features
        self.is_frozen = isinstance(adaptation, Frozen)
        self.model = adaptation.apply(backbone=model)

    def forward(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """
        Args:
            x: Waveforms with shape (B, samples).
            **kwargs: Additional backbone arguments.

        Returns:
            Raw SSL features with the backbone-specific shape.
        """
        return self.model(x, **kwargs)

    def train(self, mode: bool = True) -> "PretrainedWrapper":
        super().train(mode)
        if self.is_frozen:
            self.model.eval()
        return self
