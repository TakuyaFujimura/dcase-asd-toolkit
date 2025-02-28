from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .standard import vis_standard


def visualize_umap(umap_df: pd.DataFrame, vis_type: str, save_path: Path) -> None:
    if vis_type == "standard":
        vis_standard(umap_df=umap_df)
    else:
        raise ValueError(f"Unexpected vis_type: {vis_type}")
    plt.savefig(save_path, bbox_inches="tight")
