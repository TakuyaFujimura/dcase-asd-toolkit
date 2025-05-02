from abc import ABC, abstractmethod

import torch
from torch import nn

from .modules import AttnStatPool, add_lora


class BaseLoRA(nn.Module, ABC):
    def __init__(
        self,
        ckpt_path: str,
        lora_cfg: dict,
        embed_size: int = 128,
        last_layer: str = "linear",
        model_cfg: dict = {},
    ):
        super().__init__()
        model, feature_dim = self.construct_model(ckpt_path, **model_cfg)
        self.model = add_lora(model, lora_cfg)
        self.embed_size = embed_size
        self.last_layer = last_layer
        if self.last_layer == "linear":
            self.network = nn.Linear(feature_dim, self.embed_size)
        elif self.last_layer == "attn_stat_pool":
            self.network = nn.Sequential(
                AttnStatPool(embed_size=feature_dim),
                nn.Linear(feature_dim, self.embed_size),
            )
        else:
            raise NotImplementedError(f"last_layer={self.last_layer} is not supported.")

    @abstractmethod
    def construct_model(self, ckpt_path: str, **kwargs) -> tuple[nn.Module, int]:
        """
        Args:
            ckpt_path (str): path to the checkpoint file

        Returns:
            SSL Model[nn.Module]: ssl model
            feature dim [int]: feature dimension of the model
        """
        pass

    @abstractmethod
    def extract_features(self, x):
        pass

    def forward(self, x):
        """
        Args
            x: (B, L)
        """
        z = self.extract_features(x)

        if self.last_layer == "linear":
            z = self.network(z)  # (B, L, D)
            z = torch.mean(z, dim=1)  # (B, D)
        elif self.last_layer == "attn_stat_pool":
            z = self.network(z)  # (B, D)
        else:
            raise NotImplementedError(f"last_layer={self.last_layer} is not supported.")

        return z
