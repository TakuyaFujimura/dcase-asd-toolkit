from abc import ABC, abstractmethod
from typing import Dict

import torch
from torch import nn


class BaseBatchAugmentation(nn.Module, ABC):
    """Interface for augmentations that receive and return a batch dictionary."""

    @abstractmethod
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Return an augmented batch."""
        raise NotImplementedError
