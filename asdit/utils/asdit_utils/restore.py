import importlib
import logging
from pathlib import Path

from omegaconf import OmegaConf

from asdit.frontends.base_plmodel import BasePLFrontend
from asdit.utils.asdit_utils.path import get_version_dir
from asdit.utils.common import get_best_path, get_path_glob
from asdit.utils.config_class import DMConfig, MainExtractConfig, MainTrainConfig
from asdit.utils.dcase_utils.parse_sec import parse_sec_cfg

logger = logging.getLogger(__name__)


def get_past_cfg(cfg: MainExtractConfig) -> MainTrainConfig:
    assert cfg.restore_model_ver is not None
    config_path = (
        get_version_dir(cfg=cfg)
        / "model"
        / cfg.restore_model_ver
        / ".hydra/config.yaml"
    )

    past_cfg = MainTrainConfig(
        **parse_sec_cfg(OmegaConf.to_object(OmegaConf.load(config_path)))
    )
    return past_cfg


def get_ckpt_path(cfg: MainExtractConfig) -> Path:
    assert cfg.restore_model_ver is not None
    assert cfg.restore_ckpt_ver is not None
    ckpt_dir = (
        get_version_dir(cfg=cfg) / "model" / cfg.restore_model_ver / "checkpoints"
    )
    if cfg.restore_ckpt_ver in ["min", "max"]:
        ckpt_path = get_best_path(ckpt_dir, cfg.restore_ckpt_ver)
    elif cfg.restore_ckpt_ver == "last":
        ckpt_path = ckpt_dir / "last.ckpt"
    elif cfg.restore_ckpt_ver.startswith("epoch"):
        epoch = int(cfg.restore_ckpt_ver.split("_")[-1])
        ckpt_condition = str(ckpt_dir / f"interval_epoch={epoch-1}-*.ckpt")
        ckpt_path = Path(get_path_glob(ckpt_condition))
    return ckpt_path


def restore_plfrontend(
    cfg: MainExtractConfig, past_cfg: MainTrainConfig
) -> BasePLFrontend:
    ckpt_path = get_ckpt_path(cfg)
    logger.info(f"Loading model from {ckpt_path}")
    module_name = ".".join(past_cfg.frontend.tgt_class.split(".")[:-1])
    class_name = past_cfg.frontend.tgt_class.split(".")[-1]
    module = importlib.import_module(module_name)
    frontend_cls = getattr(module, class_name)
    frontend = frontend_cls.load_from_checkpoint(ckpt_path, strict=True)
    frontend.to(cfg.device)
    frontend.eval()
    logger.info("Model was successfully loaded from the ckpt_path")
    return frontend
