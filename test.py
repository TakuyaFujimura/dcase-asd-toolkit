# Copyright 2024 Takuya Fujimura

import logging
from pathlib import Path
from typing import Any, Dict, List, cast

import hydra
import lightning.pytorch as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from mypkgs.test_utils import evaluate_main, extract_main, score_main
from mypkgs.utils.config_class import MainTestConfig
from mypkgs.utils.config_class.main_config import MainConfig
from mypkgs.utils.io_utils import get_best_path, get_path_glob, read_json, set_logging

logger = logging.getLogger(__name__)


def hydra_to_pydantic(config: DictConfig) -> MainTestConfig:
    """Converts Hydra config to Pydantic config."""
    config_dict = cast(Dict[str, Any], OmegaConf.to_object(config))
    return MainTestConfig(**config_dict)


def get_machines(past_cfg: MainConfig) -> List[str]:
    return read_json(f"config/test/machines/{past_cfg.datamodule.dcase}.json")  # type: ignore


def get_ckpt_path(cfg: MainTestConfig) -> Path:
    ckpt_dir = cfg.result_dir / cfg.name / cfg.version / "checkpoints"
    if cfg.infer_ver == "best":
        ckpt_path = get_best_path(ckpt_dir)
    elif cfg.infer_ver == "last":
        ckpt_path = ckpt_dir / "last.ckpt"
    elif cfg.infer_ver.startswith("epoch"):
        epoch = int(cfg.infer_ver.split("_")[-1])
        ckpt_condition = str(ckpt_dir / f"interval_epoch={epoch-1}-*.ckpt")
        ckpt_path = Path(get_path_glob(ckpt_condition))
    return ckpt_path


def get_past_cfg(cfg: MainTestConfig) -> MainConfig:
    ckpt_path = get_ckpt_path(cfg)
    config_path = ckpt_path.parents[1] / ".hydra/config.yaml"
    past_cfg = MainConfig(**cast(Dict[str, Any], OmegaConf.load(config_path)))
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
    pl.seed_everything(cfg.seed, workers=True)

    infer_dir = make_dir(cfg)
    past_cfg = get_past_cfg(cfg)
    machines = get_machines(past_cfg)

    for m in machines:
        if cfg.extract:
            extract_main(cfg, infer_dir, m)
        if cfg.score:
            score_main(cfg, infer_dir, m)
        if cfg.evaluate:
            evaluate_main(cfg, infer_dir, m)


if __name__ == "__main__":
    main()
