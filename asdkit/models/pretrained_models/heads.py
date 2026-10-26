"""Projection heads accept input_dim and expose their output embedding_dim.

LinearHead consumes vectors; pooling heads consume token sequences. Select the
corresponding transform_features output mode before calling a head.
"""

from abc import ABC, abstractmethod

import torch
from torch import nn

from ..modules import AttnStatPool


class BaseHead(nn.Module, ABC):
    """Projection head with explicit input and output feature dimensions."""

    def __init__(self, input_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map features in the head-specific layout to (B, embedding_dim)."""
        pass


class LinearHead(BaseHead):
    """
    Args:
        input_dim: Width of each input feature vector.
        embedding_dim: Width of the projected embedding.
    """

    def __init__(self, input_dim: int, embedding_dim: int) -> None:
        super().__init__(input_dim=input_dim, embedding_dim=embedding_dim)
        self.projection = nn.Linear(input_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Features with shape (B, input_dim).

        Returns:
            Embeddings with shape (B, embedding_dim).
        """
        return self.projection(x)


class MeanPoolingHead(BaseHead):
    """
    Args:
        input_dim: Width of each input feature vector.
        embedding_dim: Width of the projected embedding.
    """

    def __init__(self, input_dim: int, embedding_dim: int) -> None:
        super().__init__(input_dim=input_dim, embedding_dim=embedding_dim)
        self.projection = nn.Linear(input_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Features with shape (B, tokens, input_dim).

        Returns:
            Embeddings with shape (B, embedding_dim).
        """
        return self.projection(x.mean(dim=1))


class AttnStatPoolHead(BaseHead):
    """
    Args:
        input_dim: Width of each input feature vector.
        embedding_dim: Width of the projected embedding.
    """

    def __init__(self, input_dim: int, embedding_dim: int) -> None:
        super().__init__(input_dim=input_dim, embedding_dim=embedding_dim)
        self.head = nn.Sequential(
            AttnStatPool(embedding_dim=input_dim), nn.Linear(input_dim, embedding_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Features with shape (B, tokens, input_dim).

        Returns:
            Embeddings with shape (B, embedding_dim).
        """
        return self.head(x)
