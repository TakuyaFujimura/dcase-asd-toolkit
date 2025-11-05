from abc import ABC, abstractmethod
from typing import Optional

import torch
from torch import nn

from ..modules import AttnStatPool, add_lora


class BasePretrainedWrapper(nn.Module, ABC):
    def __init__(
        self,
        embed_size: int = 128,
        projection_type: str = "linear",
        model_cfg: Optional[dict] = None,
    ):
        """
        Args:
            embed_size (int): Size of the embedding.
            projection_type (str): Type of projection layer. Options are "linear" or "attn_stat_pool".
            model_cfg (Dict[str, Any]): Configuration for the pre-trained model. Parameters in this dictionary are used in `self.construct_model`, which is defined in subclasses.
        """
        super().__init__()
        if model_cfg is None:
            model_cfg = {}
        self.projection_type = projection_type
        self.model, feature_dim = self.construct_model(**model_cfg)
        self.embed_size = embed_size
        if self.projection_type == "linear":
            self.projection_net = nn.Linear(feature_dim, self.embed_size)
        elif self.projection_type == "attn_stat_pool":
            self.projection_net = nn.Sequential(
                AttnStatPool(embed_size=feature_dim),
                nn.Linear(feature_dim, self.embed_size),
            )
        else:
            raise NotImplementedError(
                f"projection_type={self.projection_type} is not supported."
            )

    @abstractmethod
    def construct_model(self, *args, **kwargs) -> tuple[nn.Module, int]:
        """
        Args:

        Returns:
            Pretrained Model[nn.Module]
            feature dim [int]
        """
        pass

    @abstractmethod
    def extract_features(self, x) -> torch.Tensor:
        pass

    def forward(self, x):
        """
        Args
            x: (B, L)
        """
        z = self.extract_features(x)

        if self.projection_type == "linear":
            z = torch.mean(z, dim=1)  # (B, L, D) -> (B, D)
            z = self.projection_net(z)  # (B, D)
        elif self.projection_type == "attn_stat_pool":
            z = self.projection_net(z)  # (B, L, D) -> (B, D)
        else:
            raise NotImplementedError(
                f"projection_type={self.projection_type} is not supported."
            )

        return z


class BaseLoRA(BasePretrainedWrapper):
    def __init__(
        self,
        lora_cfg: dict,
        embed_size: int = 128,
        projection_type: str = "linear",
        model_cfg: Optional[dict] = None,
    ):
        super().__init__(
            embed_size=embed_size,
            projection_type=projection_type,
            model_cfg=model_cfg,
        )
        self.model = add_lora(self.model, lora_cfg)
