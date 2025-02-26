# Copyright 2024 Takuya Fujimura

import logging
from pathlib import Path
from typing import Any, Dict, List, cast

import hydra
import lightning.pytorch as pl
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from asdlib.datasets.torch_dataset import parse_path_selector
from asdlib.test_utils import score_main
from asdlib.test_utils.evaluate import evaluate_main
from asdlib.test_utils.extract import extract_main
from asdlib.test_utils.table.main import table_main
from asdlib.test_utils.umap.main import umap_main
from asdlib.utils.config_class import MainTestConfig
from asdlib.utils.config_class.main_config import MainConfig

logger = logging.getLogger(__name__)


def hydra_to_pydantic(config: DictConfig) -> MainTestConfig:
    """Converts Hydra config to Pydantic config."""
    config_dict = cast(Dict[str, Any], OmegaConf.to_object(config))
    return MainTestConfig(**config_dict)


def get_past_cfg(cfg: MainTestConfig) -> MainConfig:
    config_path = (
        cfg.result_dir / cfg.name / cfg.version / cfg.machine / ".hydra/config.yaml"
    )
    past_cfg = MainConfig(**cast(Dict[str, Any], OmegaConf.load(config_path)))
    assert cfg.result_dir == past_cfg.result_dir
    assert cfg.name == past_cfg.name
    assert cfg.version == past_cfg.version
    assert cfg.machine == past_cfg.machine
    return past_cfg


def make_dir(cfg: MainTestConfig, machine: str) -> Path:
    machine_dir = (
        cfg.result_dir / cfg.name / cfg.version / machine / "infer" / cfg.infer_ver
    )
    machine_dir.mkdir(exist_ok=True, parents=True)
    OmegaConf.save(cfg.model_dump(), machine_dir / "hparams.yaml")
    return machine_dir


def get_path_selector_list(cfg: MainTestConfig, past_cfg: MainConfig) -> List[str]:
    if cfg.path_selector_list is not None:
        return cfg.path_selector_list

    if past_cfg.machine == "_all_":
        path_selector_list = [
            f"{past_cfg.data_dir}/formatted/{past_cfg.dcase}/raw/*/train/*.wav",
            f"{past_cfg.data_dir}/formatted/{past_cfg.dcase}/raw/*/test/*.wav",
        ]
    else:
        path_selector_list = [
            f"{past_cfg.data_dir}/formatted/{past_cfg.dcase}/raw/{past_cfg.machine}/train/*.wav",  # noqa
            f"{past_cfg.data_dir}/formatted/{past_cfg.dcase}/raw/{past_cfg.machine}/test/*.wav",  # noqa
        ]

    logger.info(f"Use past path_selector_list: {path_selector_list}")
    return path_selector_list


def get_all_path(path_selector_list: List[str]) -> List[str]:
    all_path: List[str] = []

    for selector in path_selector_list:
        all_path += parse_path_selector(selector)

    if len(all_path) == 0:
        raise ValueError("No files are found")

    logger.info(f"{len(all_path)} files are found")
    return all_path


def get_machines(all_path: List[str], past_cfg: MainConfig) -> List[str]:
    machines: List[str] = []

    for p in all_path:
        # p is in the format of "<data_dir>/formatted/<dcase>/raw/<machine>/train_or_test/hoge.wav
        split_p = p.split("/")
        machines.append(split_p[-3])
        data_dir = "/".join(split_p[:-6])
        dcase = split_p[-5]
        if data_dir != past_cfg.data_dir:
            raise ValueError(f"Unmatched data_dir: {data_dir} vs {past_cfg.data_dir}")
        if dcase != past_cfg.dcase:
            raise ValueError(f"Unmatched dcase: {dcase} vs {past_cfg.dcase}")

    machines = list(set(machines))
    logger.info(f"{len(machines)} machines: {machines}")
    return machines


@hydra.main(version_base=None, config_path="../../config/test", config_name="config")
def main(hydra_cfg: DictConfig) -> None:
    cfg = hydra_to_pydantic(hydra_cfg)
    # set_logging(cfg.result_dir, __file__)
    logger.info(f"Start testing: {HydraConfig().get().run.dir}")
    logger.info(f"version: {cfg.version}")
    if cfg.tf32:
        torch.set_float32_matmul_precision("high")
        logger.info("Set float32_matmul_precision to high")
    pl.seed_everything(cfg.seed, workers=True)

    past_cfg = get_past_cfg(cfg)

    path_selector_list = get_path_selector_list(cfg=cfg, past_cfg=past_cfg)
    all_path = get_all_path(path_selector_list=path_selector_list)
    machines = get_machines(all_path=all_path, past_cfg=past_cfg)

    for m in machines:
        machine_dir = make_dir(cfg=cfg, machine=m)

        if cfg.extract:
            extract_main(
                cfg=cfg,
                past_cfg=past_cfg,
                machine_dir=machine_dir,
                all_path=all_path,
            )
        if cfg.score:
            score_main(cfg=cfg, machine_dir=machine_dir)
        if cfg.evaluate:
            evaluate_main(cfg=cfg, machine_dir=machine_dir)
        if cfg.umap:
            umap_main(cfg=cfg, machine_dir=machine_dir)


if __name__ == "__main__":
    main()
