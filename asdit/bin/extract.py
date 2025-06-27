# Copyright 2024 Takuya Fujimura

import logging
from pathlib import Path

import hydra
import lightning.pytorch as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from asdit.frontends.base import BaseFrontend
from asdit.utils.asdit_utils.extract import (
    check_cfg_with_past_cfg,
    get_extract_dataloader,
    loader2df,
)
from asdit.utils.asdit_utils.path import check_file_exists, get_output_dir
from asdit.utils.asdit_utils.restore import get_past_cfg, restore_plfrontend
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


def setup_frontend(cfg: MainExtractConfig) -> BaseFrontend:
    if cfg.restore_or_scratch == "restore":
        if cfg.restore_model_ver is None:
            raise ValueError(
                "restore_model_ver must be specified when restore_or_scratch is restore"
            )
        if cfg.restore_ckpt_ver is None:
            raise ValueError(
                "restore_ckpt_ver must be specified when restore_or_scratch is restore"
            )
        past_cfg = get_past_cfg(cfg=cfg)
        frontend = restore_plfrontend(cfg=cfg, past_cfg=past_cfg)
        check_cfg_with_past_cfg(cfg=cfg, past_cfg=past_cfg)

    elif cfg.restore_or_scratch == "scratch":
        if cfg.scratch_frontend is None:
            raise ValueError(
                "scratch_frontend must be specified when restore_or_scratch is scratch"
            )
        frontend = instantiate_tgt(cfg.scratch_frontend)
    else:
        raise ValueError(f"Unexpected restore_or_scratch: {cfg.restore_or_scratch}")

    return frontend


@hydra.main(
    version_base=None, config_path="../../config/extract", config_name="asdit_cfg"
)
def main(hydra_cfg: DictConfig) -> None:
    cfg = hydra_to_pydantic(hydra_cfg)
    logger.info(f"Start extraction: {HydraConfig().get().run.dir}")
    pl.seed_everything(seed=0, workers=True)

    output_dir = make_dir(cfg=cfg)
    frontend = setup_frontend(cfg=cfg)

    for split in ["train", "test"]:
        logger.info(f"Extracting {split} data now...")
        dataloader = get_extract_dataloader(cfg=cfg, split=split)
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
