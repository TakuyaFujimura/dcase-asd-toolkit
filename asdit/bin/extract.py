# Copyright 2024 Takuya Fujimura

import logging
from pathlib import Path

import hydra
import lightning.pytorch as pl
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from asdit.utils.asdit_utils.extract import (
    get_extract_dataloader,
    loader2dict,
    setup_frontend,
)
from asdit.utils.asdit_utils.path import check_file_exists, get_output_dir
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
        dir_path=output_dir, file_name="*_extract.npz", overwrite=cfg.overwrite
    )
    output_dir.mkdir(exist_ok=True, parents=True)
    OmegaConf.save(cfg.model_dump(), output_dir / "hparams.yaml")
    return output_dir



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
        extract_dict = loader2dict(
            dataloader=dataloader,
            frontend=frontend,
            device=cfg.device,
            extract_items=cfg.extract_items,
        )
        npz_path = output_dir / f"{split}_extract.npz"
        np.savez(npz_path, **extract_dict)
        logger.info(f"Saved extraction result to {npz_path}")


if __name__ == "__main__":
    main()
