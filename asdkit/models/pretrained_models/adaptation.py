from abc import ABC, abstractmethod
from typing import Any, Dict

from torch import nn


class PretrainedAdaptation(ABC):
    """Backbone adaptation interface."""

    @abstractmethod
    def apply(self, backbone: nn.Module) -> nn.Module:
        """
        Args:
            backbone: Pretrained SSL module.

        Returns:
            Backbone configured for the selected adaptation.
        """
        pass


class Frozen(PretrainedAdaptation):
    """Freeze parameters and keep the backbone in evaluation mode."""

    def apply(self, backbone: nn.Module) -> nn.Module:
        backbone.requires_grad_(False)
        backbone.eval()
        return backbone


class LoRA(PretrainedAdaptation):
    """
    Args:
        lora_cfg: Keyword arguments for peft.LoraConfig.
    """

    def __init__(self, lora_cfg: Dict[str, Any]) -> None:
        self.lora_cfg = lora_cfg

    def apply(self, backbone: nn.Module) -> nn.Module:
        from ..modules import add_lora

        return add_lora(model=backbone, lora_cfg=self.lora_cfg)


class FullFineTuning(PretrainedAdaptation):
    """Enable gradients for all backbone parameters."""

    def apply(self, backbone: nn.Module) -> nn.Module:
        backbone.requires_grad_(True)
        return backbone
