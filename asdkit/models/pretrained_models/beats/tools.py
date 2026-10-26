import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch

from .original.BEATs import BEATs, BEATsConfig

logger = logging.getLogger(__name__)


def restore(
    ckpt_path: Union[Path, str], update_cfg: Optional[Dict[str, Any]] = None
) -> Tuple[torch.nn.Module, int]:
    """
    Args:
        ckpt_path: Local BEATs checkpoint.
        update_cfg: Overrides for the checkpoint configuration.

    Returns:
        Restored backbone and its encoder embedding dimension.
    """
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
    if "finetuned" in ckpt_path.name:
        logger.info("Removing predictor from the finetuned BEATs model")
        model.predictor = None
    return model, cfg.encoder_embed_dim
