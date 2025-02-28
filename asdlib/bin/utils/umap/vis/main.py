import logging
from pathlib import Path

import pandas as pd

from .standard import vis_standard

logger = logging.getLogger(__name__)


def visualize_umap(umap_df: pd.DataFrame, vis_type: str, path_stem: Path) -> None:
    if vis_type == "standard":
        vis_standard(umap_df=umap_df, path_stem=path_stem)
    else:
        raise ValueError(f"Unexpected vis_type: {vis_type}")

    logger.info(f"Visualized UMAP: {f'{path_stem}_*.png'}")
