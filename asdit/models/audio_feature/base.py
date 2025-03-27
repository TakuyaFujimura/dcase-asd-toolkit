from abc import ABC, abstractmethod

import torch


class BaseAudioFeature(ABC):
    feat_dim: int

    @abstractmethod
    def __call__(self, wave: torch.Tensor) -> torch.Tensor:
        pass
