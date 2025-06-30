# Copyright 2024 Takuya Fujimura

import logging

import hydra
import lightning.pytorch as pl
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from asdit.utils.asdit_utils.extract import (
    extraction_setup_restore,
    extraction_setup_scratch,
    loader2dict,
)
from asdit.utils.asdit_utils.path import make_output_dir
from asdit.utils.config_class import MainExtractConfig
from asdit.utils.dcase_utils import parse_sec_cfg

logger = logging.getLogger(__name__)


def hydra_to_pydantic(config: DictConfig) -> MainExtractConfig:
    """Converts Hydra config to Pydantic config."""
    config_dict = parse_sec_cfg(OmegaConf.to_object(config))
    return MainExtractConfig(**config_dict)


@hydra.main(version_base=None, config_path="../../config/extract", config_name="main")
def main(hydra_cfg: DictConfig) -> None:
    cfg = hydra_to_pydantic(hydra_cfg)
    logger.info(f"Start extraction: {HydraConfig().get().run.dir}")
    pl.seed_everything(seed=0, workers=True)

    output_dir = make_output_dir(cfg, "*_extract.npz")
    if cfg.restore_or_scratch == "restore":
        frontend, dataloader_dict = extraction_setup_restore(cfg)
    elif cfg.restore_or_scratch == "scratch":
        frontend, dataloader_dict = extraction_setup_scratch(cfg)
    else:
        raise ValueError(f"Unexpected restore_or_scratch: {cfg.restore_or_scratch}")

    for split in ["train", "test"]:
        logger.info(f"Extracting {split} data now...")
        extract_dict = loader2dict(
            dataloader=dataloader_dict[split],
            frontend=frontend,
            device=cfg.device,
            extract_items=cfg.extract_items,
        )
        npz_path = output_dir / f"{split}_extract.npz"
        np.savez(npz_path, **extract_dict)
        logger.info(f"Saved extraction result to {npz_path}")


if __name__ == "__main__":
    main()
