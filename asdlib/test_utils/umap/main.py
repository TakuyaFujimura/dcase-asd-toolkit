# Copyright 2024 Takuya Fujimura

import logging
import warnings
from pathlib import Path

from ...utils.config_class.main_test_config import MainTestConfig
from .trans import get_umap_df
from .vis_main import visualize

logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    message=".*n_jobs value 1 overridden to 1 by setting random_state. Use no seed for parallelism..*",
)
warnings.filterwarnings(
    "ignore",
    message=".*'force_all_finite' was renamed to 'ensure_all_finite' in 1.6 and will be removed in 1.8..*",
)


def umap_main(
    cfg: MainTestConfig,
    machine_dir: Path,
) -> None:
    if cfg.umap_cfg is None:
        raise ValueError("`umap_cfg` is not set although `umap` is True")

    logger.info(f"Start umapping {machine_dir}")
    umap_df = get_umap_df(
        machine_dir=machine_dir,
        metric=cfg.umap_cfg.metric,
        overwrite=cfg.umap_cfg.overwrite,
        embed_key=cfg.umap_cfg.embed_key,
    )
    visualize(
        umap_df=umap_df,
        machine_dir=machine_dir,
        vis_type=cfg.umap_cfg.vis_type,
    )
