# Copyright 2024 Takuya Fujimura

import logging
from pathlib import Path
from typing import Any, Dict, cast

import hydra
import lightning.pytorch as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from asdlib.bin.utils.extract import loader2df
from asdlib.bin.utils.path import check_file_exists, get_output_dir
from asdlib.bin.utils.resume import get_past_cfg, load_plmodel
from asdlib.configmaker.extractdm import BaseExtractDMConfigMaker
from asdlib.datasets.pl_datamodule import PLDataModule
from asdlib.utils.common import instantiate_tgt
from asdlib.utils.config_class import MainExtractConfig

logger = logging.getLogger(__name__)


def hydra_to_pydantic(config: DictConfig) -> MainExtractConfig:
    """Converts Hydra config to Pydantic config."""
    config_dict = cast(Dict[str, Any], OmegaConf.to_object(config))
    return MainExtractConfig(**config_dict)


def make_dir(cfg: MainExtractConfig) -> Path:
    output_dir = get_output_dir(cfg)

    check_file_exists(
        dir_path=output_dir, file_name="*_extraction.csv", overwrite=cfg.overwrite
    )
    output_dir.mkdir(exist_ok=True, parents=True)
    OmegaConf.save(cfg.model_dump(), output_dir / "hparams.yaml")
    return output_dir


@hydra.main(version_base=None, config_path="../../config/extract", config_name="config")
def main(hydra_cfg: DictConfig) -> None:
    cfg = hydra_to_pydantic(hydra_cfg)
    logger.info(f"Start extraction: {HydraConfig().get().run.dir}")
    pl.seed_everything(seed=0, workers=True)

    output_dir = make_dir(cfg=cfg)
    past_cfg = get_past_cfg(cfg=cfg)
    DMConfigMaker: BaseExtractDMConfigMaker = instantiate_tgt(
        {**cfg.datamodule, "past_cfg": past_cfg, "machine": cfg.machine}
    )
    plmodel = load_plmodel(cfg=cfg, past_cfg=past_cfg)

    for split in ["train", "test"]:
        logger.info(f"Extracting {split} data now...")
        datamoduleconfig = DMConfigMaker.get_config(split=split)
        dataloader = PLDataModule.get_loader(
            datamoduleconfig=datamoduleconfig, label_dict_path=cfg.label_dict_path
        )
        df = loader2df(
            dataloader=dataloader,
            plmodel=plmodel,
            device=cfg.device,
            extract_items=cfg.extract_items,
        )
        df_path = output_dir / f"{split}_extraction.csv"
        df.to_csv(df_path, index=False)
        logger.info(f"Saved extraction result to {df_path}")


if __name__ == "__main__":
    main()
