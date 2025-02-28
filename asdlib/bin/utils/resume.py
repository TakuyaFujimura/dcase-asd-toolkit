import importlib
import logging
from pathlib import Path
from typing import Any, Dict, cast

from omegaconf import OmegaConf

from asdlib.pl_models import BasePLModel
from asdlib.utils.common import get_best_path, get_path_glob
from asdlib.utils.config_class import MainExtractConfig, MainTrainConfig

logger = logging.getLogger(__name__)


def get_past_cfg(cfg: MainExtractConfig) -> MainTrainConfig:
    config_path = (
        cfg.result_dir
        / cfg.name
        / f"{cfg.version}_{cfg.seed}"
        / "model"
        / cfg.model_ver
        / ".hydra/config.yaml"
    )
    past_cfg = MainTrainConfig(**cast(Dict[str, Any], OmegaConf.load(config_path)))
    return past_cfg


def get_ckpt_path(cfg: MainExtractConfig) -> Path:
    ckpt_dir = (
        cfg.result_dir
        / cfg.name
        / f"{cfg.version}_{cfg.seed}"
        / "model"
        / cfg.model_ver
        / "checkpoints"
    )
    if cfg.ckpt_ver == "best":
        ckpt_path = get_best_path(ckpt_dir)
    elif cfg.ckpt_ver == "last":
        ckpt_path = ckpt_dir / "last.ckpt"
    elif cfg.ckpt_ver.startswith("epoch"):
        epoch = int(cfg.ckpt_ver.split("_")[-1])
        ckpt_condition = str(ckpt_dir / f"interval_epoch={epoch-1}-*.ckpt")
        ckpt_path = Path(get_path_glob(ckpt_condition))
    return ckpt_path


def load_plmodel(cfg: MainExtractConfig, past_cfg: MainTrainConfig) -> BasePLModel:
    ckpt_path = get_ckpt_path(cfg)
    module_name = ".".join(past_cfg.model.tgt_class.split(".")[:-1])
    class_name = past_cfg.model.tgt_class.split(".")[-1]
    module = importlib.import_module(module_name)
    plmodel_cls = getattr(module, class_name)
    plmodel = plmodel_cls.load_from_checkpoint(ckpt_path)
    plmodel.to(cfg.device)
    plmodel.eval()
    logger.info("Model was successfully loaded from ckpt_path")
    return plmodel
