from pathlib import Path
from typing import Dict

import pandas as pd
from matplotlib import pyplot as plt

from .vis_utils import PlotConfig, get_u_idx, plot_ax_cfg


def get_cfg_dict(umap_df: pd.DataFrame) -> Dict[str, PlotConfig]:
    alpha = 0.4
    s = 5
    cfg_dict = {}
    cfg_dict["train_source_normal"] = PlotConfig(
        ax_idx=0,
        plot_cfg={
            "marker": "o",
            "alpha": alpha,
            "s": s,
            "c": "green",
        },
    )
    cfg_dict["train_target_normal"] = PlotConfig(
        ax_idx=0,
        plot_cfg={
            "marker": "o",
            "alpha": alpha,
            "s": s,
            "c": "orange",
        },
    )
    cfg_dict["test_source_normal"] = PlotConfig(
        ax_idx=1,
        plot_cfg={
            "marker": "o",
            "alpha": alpha,
            "s": s,
            "edgecolors": "green",
            "facecolor": "None",
        },
    )
    cfg_dict["test_source_anomaly"] = PlotConfig(
        ax_idx=1,
        plot_cfg={
            "marker": "x",
            "alpha": alpha,
            "s": s,
            "c": "green",
        },
    )
    cfg_dict["test_target_normal"] = PlotConfig(
        ax_idx=1,
        plot_cfg={
            "marker": "o",
            "alpha": alpha,
            "s": s,
            "edgecolors": "orange",
            "facecolor": "None",
        },
    )
    cfg_dict["test_target_anomaly"] = PlotConfig(
        ax_idx=1,
        plot_cfg={
            "marker": "x",
            "alpha": alpha,
            "s": s,
            "c": "orange",
        },
    )

    for label in cfg_dict:
        cfg_dict[label].u_idx = get_u_idx(umap_df, label)
        cfg_dict[label].plot_cfg["label"] = label
    return cfg_dict


def vis_standard(umap_df: pd.DataFrame, machine_dir: Path) -> None:
    cfg_dict = get_cfg_dict(umap_df=umap_df)
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    for cfg in cfg_dict.values():
        plot_ax_cfg(axes=axes, umap_df=umap_df, cfg=cfg)
    for i in range(2):
        axes[i].legend(bbox_to_anchor=(1, 1.1), ncol=2, loc="upper right", fontsize=15)
    plt.savefig(machine_dir / "umap_standard.png", bbox_inches="tight")
