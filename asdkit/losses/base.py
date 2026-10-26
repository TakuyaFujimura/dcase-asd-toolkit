"""Interface for classification heads used by discriminative frontends."""

from abc import ABC, abstractmethod

from torch import Tensor, nn


class BaseClassificationHead(nn.Module, ABC):
    """Expose inference logits and target-dependent training losses."""

    @abstractmethod
    def calc_logits(self, x: Tensor) -> Tensor:
        """Map embeddings (B, D) to logits (B, C) without updating head state."""
        raise NotImplementedError

    @abstractmethod
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Compute loss from embeddings (B, D) and target distributions (B, C).

        Return a scalar for mean/sum reduction or (B,) for none reduction.
        Adaptive state may be updated during training by concrete heads.
        """
        raise NotImplementedError
