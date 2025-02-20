# Copyright 2024 Takuya Fujimura

import logging
from pathlib import Path
from typing import List

from ...utils.config_class.main_test_config import UmapConfig
from .trans import get_umap_df
from .vis_main import visualize

logger = logging.getLogger(__name__)


def umap_main(
    umap_cfg: UmapConfig,
    infer_dir: Path,
    machines: List[str],
) -> None:

    for m in machines:
        logger.info(f"Start umapping {m}")
        machine_dir = infer_dir / m
        umap_df = get_umap_df(
            machine_dir=machine_dir,
            metric=umap_cfg.metric,
            overwrite=umap_cfg.overwrite,
        )
        visualize(
            umap_df=umap_df,
            machine_dir=machine_dir,
            vis_type=umap_cfg.vis_type,
        )
