import logging
from pathlib import Path

import torch

from .BEATs import BEATs, BEATsConfig

logger = logging.getLogger(__name__)


def resume(ckpt_path: Path) -> BEATs:
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. Please download it."
        )
    checkpoint = torch.load(ckpt_path)
    cfg = BEATsConfig(checkpoint["cfg"])
    model = BEATs(cfg)
    model.load_state_dict(checkpoint["model"])
    return model
