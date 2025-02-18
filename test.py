# Copyright 2024 Takuya Fujimura

import logging
from pathlib import Path
from typing import Any, Dict, cast

import hydra
import lightning.pytorch as pl
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from mypkgs.test_utils import evaluate_main, extract_main, score_main
from mypkgs.utils.config_class import MainTestConfig
from mypkgs.utils.config_class.main_config import MainConfig
from mypkgs.utils.io_utils import set_logging

logger = logging.getLogger(__name__)


def hydra_to_pydantic(config: DictConfig) -> MainTestConfig:
    """Converts Hydra config to Pydantic config."""
    config_dict = cast(Dict[str, Any], OmegaConf.to_object(config))
    return MainTestConfig(**config_dict)


def get_past_cfg(cfg: MainTestConfig) -> MainConfig:
    config_path = cfg.result_dir / cfg.name / cfg.version / ".hydra/config.yaml"
    past_cfg = MainConfig(**cast(Dict[str, Any], OmegaConf.load(config_path)))
    assert cfg.result_dir == past_cfg.result_dir
    assert cfg.name == past_cfg.name
    assert cfg.version == past_cfg.version
    return past_cfg


def make_dir(cfg: MainTestConfig) -> Path:
    infer_dir = cfg.result_dir / cfg.name / cfg.version / "infer" / cfg.infer_ver
    infer_dir.mkdir(exist_ok=True, parents=True)
    OmegaConf.save(cfg.model_dump(), infer_dir / "hparams.yaml")
    return infer_dir


@hydra.main(version_base=None, config_path="config_test", config_name="config")
def main(hydra_cfg: DictConfig) -> None:
    cfg = hydra_to_pydantic(hydra_cfg)
    set_logging(cfg.result_dir, __file__)
    logger.info(f"Start testing: {HydraConfig().get().run.dir}")
    logger.info(f"version: {cfg.version}")
    if cfg.tf32:
        torch.set_float32_matmul_precision("high")
        logger.info("Set float32_matmul_precision to high")
    pl.seed_everything(cfg.seed, workers=True)

    infer_dir = make_dir(cfg)
    past_cfg = get_past_cfg(cfg)

    if cfg.extract:
        extract_main(cfg=cfg, past_cfg=past_cfg, infer_dir=infer_dir)
    if cfg.score:
        score_main(cfg, infer_dir)
    if cfg.evaluate:
        evaluate_main(cfg, infer_dir)


if __name__ == "__main__":
    main()
