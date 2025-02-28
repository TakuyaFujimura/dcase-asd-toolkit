from pathlib import Path
from typing import Dict, List

import pandas as pd
from matplotlib import pyplot as plt

from .utils import get_u_idx, plot_ax_cfg


def get_cfg_list_of_dict(umap_df: pd.DataFrame) -> List[Dict[str, dict]]:
    alpha = 1.0
    s = 70
    linewidths = 2
    cfg_list_of_dict: List[dict] = [{}, {}]
    cfg_list_of_dict[0]["train_source_normal"] = {
        "marker": ",",
        "alpha": alpha,
        "s": s,
        "edgecolors": "green",
        "facecolor": "None",
        "linewidths": linewidths,
    }
    cfg_list_of_dict[0]["train_target_normal"] = {
        "marker": ",",
        "alpha": alpha,
        "s": s,
        "edgecolors": "mediumseagreen",
        "facecolor": "None",
        "linewidths": linewidths,
    }
    cfg_list_of_dict[1]["train_source_normal"] = {
        "marker": ",",
        "alpha": 0.2,
        "s": s,
        "edgecolors": "green",
        "facecolor": "None",
    }
    cfg_list_of_dict[1]["train_target_normal"] = {
        "marker": ",",
        "alpha": 0.2,
        "s": s,
        "edgecolors": "mediumseagreen",
        "facecolor": "None",
    }
    cfg_list_of_dict[1]["test_source_normal"] = {
        "marker": "o",
        "alpha": alpha,
        "s": s,
        "edgecolors": "royalblue",
        "facecolor": "None",
        "linewidths": linewidths,
    }
    cfg_list_of_dict[1]["test_source_anomaly"] = {
        "marker": "^",
        "alpha": alpha,
        "s": s,
        "edgecolors": "mediumvioletred",
        "facecolor": "None",
        "linewidths": linewidths,
    }
    cfg_list_of_dict[1]["test_target_normal"] = {
        "marker": "o",
        "alpha": alpha,
        "s": s,
        "edgecolors": "orange",
        "facecolor": "None",
        "linewidths": linewidths,
    }
    cfg_list_of_dict[1]["test_target_anomaly"] = {
        "marker": "^",
        "alpha": alpha,
        "s": s,
        "edgecolors": "red",
        "facecolor": "None",
        "linewidths": linewidths,
    }

    for i in range(len(cfg_list_of_dict)):
        for label in cfg_list_of_dict[i].keys():
            cfg_list_of_dict[i][label]["u_idx"] = get_u_idx(umap_df, label)
            cfg_list_of_dict[i][label]["label"] = label
    return cfg_list_of_dict


def plot_and_save(
    umap_df: pd.DataFrame, save_path: Path, u0min, u0max, u1min, u1max
) -> None:
    cfg_list_of_dict = get_cfg_list_of_dict(umap_df=umap_df)
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    for i in range(len(cfg_list_of_dict)):
        for cfg in cfg_list_of_dict[i].values():
            plot_ax_cfg(ax=axes[i], umap_df=umap_df, cfg=cfg)
    for i in range(2):
        axes[i].legend(bbox_to_anchor=(1, 1.1), ncol=2, loc="upper right", fontsize=15)
        axes[i].set_xlim(u0min - 0.5, u0max + 0.5)
        axes[i].set_ylim(u1min - 0.5, u1max + 0.5)
    plt.savefig(save_path, bbox_inches="tight")


def vis_standard(umap_df: pd.DataFrame, output_dir: Path, png_stem: str) -> None:
    ulim_dict = {
        "u0min": umap_df["u0"].min(),
        "u0max": umap_df["u0"].max(),
        "u1min": umap_df["u1"].min(),
        "u1max": umap_df["u1"].max(),
    }
    plot_and_save(
        umap_df=umap_df,
        save_path=output_dir / f"{png_stem}_all.png",
        **ulim_dict,
    )
    for section in umap_df["section"].unique():
        umap_df_selected = umap_df[umap_df["section"] == section]
        plot_and_save(
            umap_df=umap_df_selected,
            save_path=output_dir / f"{png_stem}_{section}.png",
            **ulim_dict,
        )
