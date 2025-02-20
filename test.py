# Copyright 2024 Takuya Fujimura

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import hydra
import lightning.pytorch as pl
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from mypkgs.datasets.torch_dataset import parse_path_selector
from mypkgs.test_utils import evaluate_main, extract_main, score_main
from mypkgs.test_utils.table.main import table_main
from mypkgs.test_utils.umap.main import umap_main
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


def get_path_selector_list(cfg: MainTestConfig, past_cfg: MainConfig) -> List[str]:
    if cfg.path_selector_list is None:
        path_selector_list = [
            f"{past_cfg.data_dir}/{past_cfg.dcase}/all/raw/*/train/*.wav",
            f"{past_cfg.data_dir}/{past_cfg.dcase}/all/raw/*/test/*.wav",
        ]
        logger.info(f"Use past path_selector_list: {path_selector_list}")
    else:
        path_selector_list = cfg.path_selector_list
    return path_selector_list


def get_all_path(path_selector_list: List[str]) -> List[str]:
    all_path: List[str] = []

    for selector in path_selector_list:
        all_path += parse_path_selector(selector)
    logger.info(f"{len(all_path)} files are found")
    return all_path


def get_machines(all_path: List[str], past_cfg: MainConfig) -> List[str]:
    machines: List[str] = []

    for p in all_path:
        # p is in the format of "<data_dir>/<dcase>/all/raw/<machine>/train-test/hoge.wav
        split_p = p.split("/")
        machines.append(split_p[-3])
        data_dir = "/".join(split_p[:-6])
        dcase = split_p[-6]
        if data_dir != past_cfg.data_dir:
            logger.warning(f"Unmatched data_dir: {data_dir} vs {past_cfg.data_dir}")
        if dcase != past_cfg.dcase:
            logger.warning(f"Unmatched dcase: {dcase} vs {past_cfg.dcase}")

    machines = list(set(machines))
    logger.info(f"{len(machines)} machines: {machines}")
    return machines


@hydra.main(version_base=None, config_path="config/test", config_name="config")
def main(hydra_cfg: DictConfig) -> None:
    cfg = hydra_to_pydantic(hydra_cfg)
    # set_logging(cfg.result_dir, __file__)
    logger.info(f"Start testing: {HydraConfig().get().run.dir}")
    logger.info(f"version: {cfg.version}")
    if cfg.tf32:
        torch.set_float32_matmul_precision("high")
        logger.info("Set float32_matmul_precision to high")
    pl.seed_everything(cfg.seed, workers=True)

    infer_dir = make_dir(cfg)
    past_cfg = get_past_cfg(cfg)
    path_selector_list = get_path_selector_list(cfg=cfg, past_cfg=past_cfg)
    all_path = get_all_path(path_selector_list=path_selector_list)
    machines = get_machines(all_path=all_path, past_cfg=past_cfg)

    if cfg.extract:
        extract_main(
            cfg=cfg,
            past_cfg=past_cfg,
            infer_dir=infer_dir,
            all_path=all_path,
            machines=machines,
        )
    if cfg.score:
        score_main(cfg=cfg, infer_dir=infer_dir, machines=machines)
    if cfg.evaluate:
        evaluate_main(cfg=cfg, infer_dir=infer_dir, machines=machines)
    if len(cfg.table_metric_list) > 0:
        table_main(
            metric_list=cfg.table_metric_list, infer_dir=infer_dir, machines=machines
        )
    if cfg.umap:
        if cfg.umap_cfg is None:
            raise ValueError("umap_cfg is not set")
        umap_main(umap_cfg=cfg.umap_cfg, infer_dir=infer_dir, machines=machines)


if __name__ == "__main__":
    main()
