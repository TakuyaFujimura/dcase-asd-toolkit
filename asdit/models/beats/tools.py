import logging
from pathlib import Path
from typing import Optional, Tuple

import torch

from .BEATs import BEATs, BEATsConfig

logger = logging.getLogger(__name__)


def restore(
    ckpt_path: Path | str, update_cfg: Optional[dict] = None
) -> Tuple[torch.nn.Module, int]:
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. Please download it."
        )
    checkpoint = torch.load(ckpt_path)
    cfg = BEATsConfig(checkpoint["cfg"])
    if update_cfg is None:
        update_cfg = {}
    cfg.update(update_cfg)
    model = BEATs(cfg)
    model.load_state_dict(checkpoint["model"])
    return model, 768
