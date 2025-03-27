import logging
from pathlib import Path

import torch

from asdit.models.beats import resume
from asdit.utils.config_class.output_config import PLOutput

from ..base import BaseFrontend

logger = logging.getLogger(__name__)


class BeatsPoolModel(BaseFrontend):
    def __init__(self, ckpt_path: str, layer: str = "last"):
        self.model = resume(Path(ckpt_path))
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.layer = layer
        if self.layer != "last":
            raise NotImplementedError("Only last layer is supported for now")

    def extract(self, batch: dict) -> PLOutput:
        x = batch["wave"]

        if self.device != x.device:
            logger.info("Move Model to the same device")
            self.device = x.device
            self.model.to(self.device)

        with torch.no_grad():
            z = self.model.extract_features(x)[0]
            z = z.mean(1)  # (B, L, C) -> (B, C)
        embed_dict = {"main": z}
        return PLOutput(embed=embed_dict)
