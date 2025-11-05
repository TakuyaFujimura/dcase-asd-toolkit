import logging
from typing import Dict, List

import numpy as np
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


def get_u_idx(
    is_test: np.ndarray, is_target: np.ndarray, is_normal: np.ndarray, label: str
) -> np.ndarray:
    label_split = label.split("_")
    # train/test
    if label_split[0] == "train":
        u_idx = is_test == 0
    elif label_split[0] == "test":
        u_idx = is_test == 1
    else:
        raise ValueError(f"Unexpected label: {label}")
    # source/target
    if label_split[1] == "source":
        u_idx = u_idx & (is_target == 0)
    elif label_split[1] == "target":
        u_idx = u_idx & (is_target == 1)
    else:
        raise ValueError(f"Unexpected label: {label}")
    # normal/anomaly
    if label_split[2] == "normal":
        u_idx = u_idx & (is_normal == 1)
    elif label_split[2] == "anomaly":
        u_idx = u_idx & (is_normal == 0)
    else:
        raise ValueError(f"Unexpected label: {label}")

    return u_idx  # type: ignore


def get_cfg_list_of_dict(
    is_test: np.ndarray, is_target: np.ndarray, is_normal: np.ndarray
) -> List[Dict[str, dict]]:
    alpha = 1.0
    s = 70
    linewidths = 2
    cfg_list_of_dict: List[dict] = [{}, {}]
    # axes[0]
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
    # axes[1]
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
            cfg_list_of_dict[i][label]["u_idx"] = get_u_idx(
                is_test=is_test, is_target=is_target, is_normal=is_normal, label=label
            )
            cfg_list_of_dict[i][label]["label"] = label
    return cfg_list_of_dict


def plot_ax_cfg(ax, umap_embed, cfg: dict):
    u_idx = cfg.pop("u_idx", None)
    ax.scatter(
        umap_embed[u_idx, 0],
        umap_embed[u_idx, 1],
        **cfg,
    )


def plot_and_save(
    umap_embed: np.ndarray,
    is_test: np.ndarray,
    is_target: np.ndarray,
    is_normal: np.ndarray,
    save_path: str,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> None:
    cfg_list_of_dict = get_cfg_list_of_dict(
        is_test=is_test, is_target=is_target, is_normal=is_normal
    )
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    for i in range(len(cfg_list_of_dict)):
        for cfg in cfg_list_of_dict[i].values():
            plot_ax_cfg(ax=axes[i], umap_embed=umap_embed, cfg=cfg)
    for i in range(2):
        axes[i].legend(bbox_to_anchor=(1, 1.15), ncol=2, loc="upper right", fontsize=15)
        axes[i].set_xlim(xmin - 0.5, xmax + 0.5)
        axes[i].set_ylim(ymin - 0.5, ymax + 0.5)
    plt.savefig(save_path, bbox_inches="tight")


def plot_umap(visualize_dict: dict, save_dir, save_path_stem) -> None:
    umap_embed = visualize_dict["umap_embed"]
    is_test = visualize_dict["is_test"]
    is_target = visualize_dict["is_target"]
    is_normal = visualize_dict["is_normal"]
    section = visualize_dict["section"]

    lim_dict = {
        "xmin": umap_embed[:, 0].min(),
        "xmax": umap_embed[:, 0].max(),
        "ymin": umap_embed[:, 1].min(),
        "ymax": umap_embed[:, 1].max(),
    }
    plot_and_save(
        umap_embed=umap_embed,
        is_test=is_test,
        is_target=is_target,
        is_normal=is_normal,
        save_path=f"{save_dir}/{save_path_stem}_sectionall.png",
        **lim_dict,
    )
    for sec in np.unique(section):
        idx = section == sec
        plot_and_save(
            umap_embed=umap_embed[idx],
            is_test=is_test[idx],
            is_target=is_target[idx],
            is_normal=is_normal[idx],
            save_path=f"{save_dir}/{save_path_stem}_section{sec}.png",
            **lim_dict,
        )
