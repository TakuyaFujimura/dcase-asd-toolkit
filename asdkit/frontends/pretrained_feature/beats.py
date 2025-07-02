import logging
from typing import Any, Dict, Optional

from torch import nn

from asdkit.models.pretrained_models.beats import restore

from .base import BaseFrozenModel

logger = logging.getLogger(__name__)


class BEATsFrozenModel(BaseFrozenModel):
    def __init__(self, model_cfg: Optional[dict] = None):
        super().__init__(model_cfg=model_cfg)

    def construct_model(
        self,
        ckpt_path: str,
        update_cfg: Optional[dict] = None,
    ) -> nn.Module:
        model, _ = restore(ckpt_path=ckpt_path, update_cfg=update_cfg)
        return model

    def extract(self, batch: dict) -> Dict[str, Any]:
        x = batch["wave"]

        if self.device != x.device:
            logger.info("Move Model to the same device")
            self.device = x.device
            self.model.to(self.device)

        z = self.model.extract_features(x)[0]
        z = z.mean(1)  # (B, L, D) -> (B, D)
        return {"embed": z}
