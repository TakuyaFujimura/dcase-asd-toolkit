from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from asdlib.utils.config_class.nparray_class import Np1DArray
from pydantic import BaseModel


class PlotConfig(BaseModel):
    u_idx: Optional[Np1DArray] = None
    plot_cfg: Dict[str, Any]


def plot_ax_cfg(ax, umap_df: pd.DataFrame, cfg: dict):
    u_idx = cfg.pop("u_idx", None)
    ax.scatter(
        umap_df["u0"][u_idx == 1],
        umap_df["u1"][u_idx == 1],
        **cfg,
    )


def get_u_idx(umap_df: pd.DataFrame, label: str) -> np.ndarray:
    label_split = label.split("_")
    # train/test
    if label_split[0] == "train":
        u_idx = umap_df["is_test"] == 0
    elif label_split[0] == "test":
        u_idx = umap_df["is_test"] == 1
    else:
        raise ValueError(f"Unexpected label: {label}")
    # source/target
    if label_split[1] == "source":
        u_idx = u_idx & (umap_df["is_target"] == 0)
    elif label_split[1] == "target":
        u_idx = u_idx & (umap_df["is_target"] == 1)
    else:
        raise ValueError(f"Unexpected label: {label}")
    # normal/anomaly
    if label_split[2] == "normal":
        u_idx = u_idx & (umap_df["is_normal"] == 1)
    elif label_split[2] == "anomaly":
        u_idx = u_idx & (umap_df["is_normal"] == 0)
    else:
        raise ValueError(f"Unexpected label: {label}")

    return u_idx  # type: ignore
