# Copyright 2024 Takuya Fujimura

import logging
import warnings
from typing import Any, Dict, cast

import hydra
import lightning.pytorch as pl
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from asdlib.bin.utils.path import check_file_exists, get_output_dir
from asdlib.bin.utils.umap import trans_umap, visualize_umap
from asdlib.utils.config_class import MainUmapConfig

logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    message=".*n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism..*",
)
warnings.filterwarnings(
    "ignore",
    message=".*'force_all_finite' was renamed to 'ensure_all_finite' in 1.6 and will be removed in 1.8..*",
)


def hydra_to_pydantic(config: DictConfig) -> MainUmapConfig:
    """Converts Hydra config to Pydantic config."""
    config_dict = cast(Dict[str, Any], OmegaConf.to_object(config))
    return MainUmapConfig(**config_dict)


@hydra.main(version_base=None, config_path="../../config/umap", config_name="config")
def main(hydra_cfg: DictConfig) -> None:
    cfg = hydra_to_pydantic(hydra_cfg)
    logger.info(f"Start scoring: {HydraConfig().get().run.dir}")
    pl.seed_everything(seed=0, workers=True)

    output_dir = get_output_dir(cfg=cfg)
    umap_dir = output_dir / "umap"
    umap_dir.mkdir(parents=True, exist_ok=True)

    csv_stem = f"umap_{cfg.metric}_{cfg.embed_key}"
    png_stem = f"umap_{cfg.metric}_{cfg.embed_key}_{cfg.vis_type}"

    if cfg.trans_exec:
        check_file_exists(
            dir_path=umap_dir,
            file_name=f"{csv_stem}.csv",
            overwrite=cfg.trans_overwrite,
        )
        trans_umap(
            output_dir=output_dir,
            metric=cfg.metric,
            embed_key=cfg.embed_key,
            save_path=umap_dir / f"{csv_stem}.csv",
            extract_items=cfg.extract_items,
        )

    if cfg.vis_exec:
        check_file_exists(
            dir_path=umap_dir,
            file_name=f"{png_stem}_*.png",
            overwrite=cfg.vis_overwrite,
        )
        umap_df = pd.read_csv(umap_dir / f"{csv_stem}.csv")

        visualize_umap(
            umap_df=umap_df,
            vis_type=cfg.vis_type,
            path_stem=umap_dir / png_stem,
        )


if __name__ == "__main__":
    main()
