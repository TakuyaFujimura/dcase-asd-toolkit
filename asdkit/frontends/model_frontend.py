from typing import Any, Dict

from torch import nn

from asdkit.utils.common import instantiate

from .base import BaseFrontend


class ModelFrontend(nn.Module, BaseFrontend):
    """
    Args:
        model_cfg: Waveform-to-embedding module configuration with _target_.
    """

    def __init__(self, model_cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.model = instantiate(model_cfg)

    def get_metadata(self) -> Dict[str, Any]:
        if hasattr(self.model, "embedding_representation"):
            return {
                "embeddings": {
                    "embed": {"representation": self.model.embedding_representation}
                }
            }
        return {}

    def extract(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return {"embed": self.model(batch["wave"])}
