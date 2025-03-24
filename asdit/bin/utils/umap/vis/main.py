import logging
from pathlib import Path

import pandas as pd

from .standard import vis_standard

logger = logging.getLogger(__name__)


def visualize_umap(
    umap_df: pd.DataFrame, save_path: Path, vis_type: str = "standard"
) -> None:
    if vis_type == "standard":
        vis_standard(umap_df=umap_df, save_path=save_path)
    else:
        logger.error(f"Unexpected vis_type: {vis_type}")
        raise ValueError(f"Unexpected vis_type: {vis_type}")

    logger.info(f"Visualized UMAP: {save_path}")
