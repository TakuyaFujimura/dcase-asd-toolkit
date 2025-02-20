from pathlib import Path

import pandas as pd

from .vis_standard import vis_standard


def visualize(umap_df: pd.DataFrame, machine_dir: Path, vis_type: str) -> None:
    if vis_type == "standard":
        vis_standard(umap_df, machine_dir)
    else:
        raise ValueError(f"Unexpected vis_type: {vis_type}")
