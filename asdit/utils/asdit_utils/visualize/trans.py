import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import umap

from asdit.utils.common.match import re_match_any

from .utils import load_npz_dict

logger = logging.getLogger(__name__)


def get_embed_is_test(
    train_dict: dict, test_dict: dict, embed_key: str
) -> Tuple[np.ndarray, np.ndarray]:
    train_embed = train_dict[embed_key]
    test_embed = test_dict[embed_key]
    embed = np.concatenate([train_embed, test_embed], axis=0)
    is_test = np.array([0] * len(train_embed) + [1] * len(test_embed))
    return embed, is_test


def get_visualize_dict(
    train_dict: dict, test_dict: dict, extract_items: List[str]
) -> dict:
    if set(train_dict.keys()) != set(test_dict.keys()):
        raise ValueError(
            "train_extract.npz and test_extract.npz must have the same keys."
        )

    visualize_dict = {}
    for key in train_dict.keys():
        if re_match_any(patterns=extract_items, string=key):
            visualize_dict[key] = np.concatenate(
                [
                    train_dict[key],
                    test_dict[key],
                ],
                axis=0,
            )
    return visualize_dict


def trans_umap(
    output_dir: Path,
    metric: str,
    embed_key: str,
    save_path: Path,
    extract_items: List[str],
) -> None:
    # Load data
    train_dict = load_npz_dict(output_dir / "train_extract.npz")
    test_dict = load_npz_dict(output_dir / "test_extract.npz")
    embed, is_test = get_embed_is_test(
        train_dict=train_dict, test_dict=test_dict, embed_key=embed_key
    )
    visualize_dict = get_visualize_dict(
        train_dict=train_dict, test_dict=test_dict, extract_items=extract_items
    )

    # accept "euclid" as an alias for "euclidean"
    if metric == "euclid":
        metric = "euclidean"

    # UMAP transformation
    umap_model = umap.UMAP(random_state=0, metric=metric)
    umap_embed = umap_model.fit_transform(embed)  # (N, 2)

    # Save
    visualize_dict["is_test"] = is_test
    visualize_dict["umap_embed"] = umap_embed
    for key in ["section", "is_normal", "is_target"]:
        if key not in visualize_dict:
            raise ValueError(
                f"The key '{key}' must be included in extract_items for visualization. "
                + "Please update the extract_items in the config file."
            )
    np.savez(save_path, **visualize_dict)
    logger.info(f"Saved: {save_path}")
