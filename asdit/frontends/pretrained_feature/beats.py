import logging
from typing import Optional

from torch import nn

from asdit.models.beats import restore
from asdit.utils.config_class.output_config import FrontendOutput

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

    def extract(self, batch: dict) -> FrontendOutput:
        x = batch["wave"]

        if self.device != x.device:
            logger.info("Move Model to the same device")
            self.device = x.device
            self.model.to(self.device)

        z = self.model.extract_features(x)[0]
        z = z.mean(1)  # (B, L, D) -> (B, D)
        embed_dict = {"main": z}
        return FrontendOutput(embed=embed_dict)
