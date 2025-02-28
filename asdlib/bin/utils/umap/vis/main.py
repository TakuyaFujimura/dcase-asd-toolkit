import logging
from pathlib import Path

import pandas as pd

from .standard import vis_standard

logger = logging.getLogger(__name__)


def visualize_umap(
    umap_df: pd.DataFrame, vis_type: str, output_dir: Path, png_stem: str
) -> None:
    if vis_type == "standard":
        vis_standard(umap_df=umap_df, output_dir=output_dir, png_stem=png_stem)
    else:
        raise ValueError(f"Unexpected vis_type: {vis_type}")

    logger.info(f"Visualized UMAP: {output_dir / f'{png_stem}_*.png'}")
