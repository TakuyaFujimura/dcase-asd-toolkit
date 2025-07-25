import logging
from abc import abstractmethod
from typing import Any, Dict, Optional

from torch import nn

from ..base import BaseFrontend

logger = logging.getLogger(__name__)


class BaseFrozenModel(BaseFrontend):
    def __init__(self, model_cfg: Optional[dict] = None):
        """
        Args:
            model_cfg (Dict[str, Any]): Configuration for the model. Parameters in this dictionary are used in `self.construct_model`, which is defined in subclasses.
        """
        if model_cfg is None:
            model_cfg = {}
        self.model = self.construct_model(**model_cfg)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        self.device = next(self.model.parameters()).device

    @abstractmethod
    def construct_model(self, *args, **kwargs) -> nn.Module:
        pass

    @abstractmethod
    def extract(self, batch: dict) -> Dict[str, Any]:
        pass
