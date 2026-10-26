import logging
import re
from pathlib import Path
from typing import Union

import torch

from asdkit.frontends.base_plmodel import BasePLFrontend
from asdkit.utils.asdkit_utils.path import get_version_dir
from asdkit.utils.common import get_best_path, get_path_glob, instantiate
from asdkit.utils.config_class import MainExtractConfig

logger = logging.getLogger(__name__)


def get_ckpt_path(cfg: MainExtractConfig) -> Path:
    """Resolve a checkpoint selector or checkpoint name independently of infer_ver."""
    if cfg.frontend_checkpoint is None:
        raise ValueError("frontend_checkpoint must be specified")
    if cfg.model_ver is None:
        raise ValueError("model_ver must be specified")
    checkpoint = cfg.frontend_checkpoint
    ckpt_dir = get_version_dir(cfg=cfg) / "model" / cfg.model_ver / "checkpoints"
    if checkpoint in ["min", "max"]:
        ckpt_path = get_best_path(ckpt_dir=ckpt_dir, best_type=checkpoint)
    elif re.fullmatch(r"epoch_[0-9]+", checkpoint):
        epoch = int(checkpoint.split("_")[-1])
        ckpt_condition = str(ckpt_dir / f"interval_epoch={epoch-1}-*.ckpt")
        ckpt_path = Path(get_path_glob(ckpt_condition))
    else:
        ckpt_path = ckpt_dir / f"{checkpoint}.ckpt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    logger.info(f"Resolved frontend checkpoint: {ckpt_path}")
    return ckpt_path


def restore_plfrontend(
    ckpt_path: Union[str, Path], device: str = "cpu", weights_only: bool = True
) -> BasePLFrontend:
    """Restore a frontend from the configuration and weights in its checkpoint."""
    ckpt_path = Path(ckpt_path)
    logger.info(f"Loading model from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=weights_only)
    frontend: BasePLFrontend = instantiate(cfg=checkpoint["frontend_cfg"])
    frontend.checkpoint_frontend_cfg = checkpoint["frontend_cfg"]
    frontend.on_load_checkpoint(checkpoint)
    frontend.load_state_dict(
        checkpoint["state_dict"], strict=not frontend.save_only_trainable
    )
    frontend.to(device)
    frontend.eval()
    logger.info("Model was successfully loaded from the ckpt_path")
    return frontend
