# Copyright 2024 Takuya Fujimura

import logging
from pathlib import Path
from typing import Tuple

import hydra
import lightning.pytorch as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from asdit.datasets.pl_datamodule import PLDataModule
from asdit.frontends.base import BaseFrontend
from asdit.utils.asdit_utils.extract import loader2df
from asdit.utils.asdit_utils.path import check_file_exists, get_output_dir
from asdit.utils.asdit_utils.restore import (
    ExtractDMConfigMaker,
    get_past_cfg,
    restore_dmconfigmaker,
    restore_plfrontend,
)
from asdit.utils.common import instantiate_tgt
from asdit.utils.config_class import MainExtractConfig
from asdit.utils.dcase_utils import parse_sec_cfg

logger = logging.getLogger(__name__)


def hydra_to_pydantic(config: DictConfig) -> MainExtractConfig:
    """Converts Hydra config to Pydantic config."""
    config_dict = parse_sec_cfg(OmegaConf.to_object(config))
    return MainExtractConfig(**config_dict)


def make_dir(cfg: MainExtractConfig) -> Path:
    output_dir = get_output_dir(cfg)

    check_file_exists(
        dir_path=output_dir, file_name="*_extract.csv", overwrite=cfg.overwrite
    )
    output_dir.mkdir(exist_ok=True, parents=True)
    OmegaConf.save(cfg.model_dump(), output_dir / "hparams.yaml")
    return output_dir


def setup_frontend_dmconfigmaker(
    cfg: MainExtractConfig,
) -> Tuple[BaseFrontend, ExtractDMConfigMaker]:
    if cfg.restore_or_scratch == "restore":
        # check
        if cfg.model_ver is None:
            raise ValueError(
                "model_ver must be specified when restore_or_scratch is restore"
            )
        if cfg.ckpt_ver is None:
            raise ValueError(
                "ckpt_ver must be specified when restore_or_scratch is restore"
            )
        past_cfg = get_past_cfg(cfg=cfg)
        frontend = restore_plfrontend(cfg=cfg, past_cfg=past_cfg)
        dmconfigmaker = restore_dmconfigmaker(cfg=cfg, past_cfg=past_cfg)

    elif cfg.restore_or_scratch == "scratch":
        if cfg.frontend_cfg is None:
            raise ValueError(
                "frontend_cfg must be specified when restore_or_scratch is scratch"
            )
        frontend = instantiate_tgt(cfg.frontend_cfg)
        dmconfigmaker = ExtractDMConfigMaker(
            dcase=cfg.dcase, machine=cfg.machine, **cfg.datamodule
        )
    else:
        raise ValueError(f"Unexpected restore_or_scratch: {cfg.restore_or_scratch}")
    return frontend, dmconfigmaker


@hydra.main(
    version_base=None, config_path="../../config/extract", config_name="asdit_cfg"
)
def main(hydra_cfg: DictConfig) -> None:
    cfg = hydra_to_pydantic(hydra_cfg)
    logger.info(f"Start extraction: {HydraConfig().get().run.dir}")
    pl.seed_everything(seed=0, workers=True)

    output_dir = make_dir(cfg=cfg)
    frontend, dmconfigmaker = setup_frontend_dmconfigmaker(cfg=cfg)

    for split in ["train", "test"]:
        logger.info(f"Extracting {split} data now...")
        datamoduleconfig = dmconfigmaker.get_config(split=split)
        dataloader = PLDataModule.get_loader(
            datamoduleconfig=datamoduleconfig, label_dict_path=cfg.label_dict_path
        )
        df = loader2df(
            dataloader=dataloader,
            frontend=frontend,
            device=cfg.device,
            extract_items=cfg.extract_items,
        )
        df_path = output_dir / f"{split}_extract.csv"
        df.to_csv(df_path, index=False)
        logger.info(f"Saved extraction result to {df_path}")


if __name__ == "__main__":
    main()
