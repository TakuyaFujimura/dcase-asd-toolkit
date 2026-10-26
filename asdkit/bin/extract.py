# Copyright 2024 Takuya Fujimura

import logging

import hydra
import lightning.pytorch as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from asdkit.utils.asdkit_utils.extract import (
    NpyExtractionWriter,
    create_extraction_frontend,
    create_extraction_loaders,
    extract_batches,
    get_num_samples,
)
from asdkit.utils.asdkit_utils.path import get_extraction_marker, get_output_dir
from asdkit.utils.common import write_json
from asdkit.utils.common.hydra import hydra_to_pydantic
from asdkit.utils.config_class import MainExtractConfig
from asdkit.utils.dcase_utils import parse_sec_cfg

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../config/extract", config_name="main")
def main(hydra_cfg: DictConfig) -> None:
    cfg: MainExtractConfig = hydra_to_pydantic(
        hydra_cfg=hydra_cfg, cls=MainExtractConfig, transform=parse_sec_cfg
    )
    logger.info(f"Start extraction: {HydraConfig().get().run.dir}")
    pl.seed_everything(seed=0, workers=True)

    output_dir = get_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    pending_splits = {}
    for split, dm_cfg in cfg.datamodule.items():
        marker = get_extraction_marker(output_dir / f"{split}_extract")
        if marker.is_file() and not cfg.overwrite:
            logger.info(f"Already completed: {split}_extract")
        else:
            pending_splits[split] = dm_cfg
    if not pending_splits:
        return
    frontend = create_extraction_frontend(cfg)
    dataloader_dict = create_extraction_loaders(pending_splits)

    for split, dataloader in dataloader_dict.items():
        logger.info(f"Extracting {split} data now...")
        writer = NpyExtractionWriter(
            output_dir / f"{split}_extract",
            num_samples=get_num_samples(dataloader),
        )
        try:
            extract_batches(
                dataloader=dataloader,
                frontend=frontend,
                device=cfg.device,
                extract_items=cfg.extract_items,
                writer=writer,
            )
            metadata_path = output_dir / f"{split}_extract.metadata.json"
            write_json(json_path=metadata_path, data=frontend.get_metadata())
            writer.finish()
            logger.info(f"Saved extraction result to {writer.directory}")
        finally:
            writer.close()


if __name__ == "__main__":
    main()
