import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from asdkit.datasets.collators import get_dcase_path
from asdkit.utils.asdkit_utils.extract import load_extract_data


@dataclass(frozen=True)
class ExtractData:
    embed: np.ndarray
    path: List[str]


def get_all_machine_list(dcase: str) -> List[str]:
    from asdkit.utils.dcase_utils import MACHINE_DICT

    all_machine_list = list(
        set(MACHINE_DICT[f"{dcase}-dev"] + MACHINE_DICT[f"{dcase}-eval"])
    )
    all_machine_list.sort()
    return all_machine_list


def get_teacher_dir(results_root_dir: Path, dcase: str, teacher_dir: Path) -> Path:
    return results_root_dir / dcase / teacher_dir


def get_output_json_path(
    labels_root_dir: Path, dcase: str, output_label_name: str
) -> Path:
    json_path = labels_root_dir / dcase / f"{output_label_name}.json"
    if json_path.exists():
        raise FileExistsError(f"{json_path} already exists.")
    return json_path


def load_extract(
    teacher_dir: Path,
    machine: str,
    embed_key: str,
) -> ExtractData:
    embed, paths = load_clustering_features(teacher_dir / machine, embed_key)
    return ExtractData(embed=embed, path=paths)


def load_clustering_features(
    output_dir: Path, embed_key: str
) -> Tuple[np.ndarray, List[str]]:
    """Read a selected feature from completed NPY training extraction."""
    data = load_extract_data(output_dir, "train").extract_dict
    embed, paths = data[embed_key], data["path"].tolist()
    if embed.ndim == 3 and embed.shape[1] == 1:
        embed = embed[:, 0, :]
    if embed.ndim != 2 or len(embed) == 0:
        raise ValueError(
            f"Clustering requires nonempty (samples, features), got {embed.shape} "
            f"for {embed_key!r} in {output_dir}"
        )
    if len(embed) != len(paths) or not all(isinstance(p, str) for p in paths):
        raise ValueError(f"Feature rows and string paths must agree in {output_dir}")
    return embed, paths


def generate_pseudo_labels(
    idx_array: np.ndarray, path_list: List[str]
) -> Dict[str, Any]:
    if len(idx_array) != len(path_list):
        raise ValueError("Labels and paths must have the same length")
    if len(idx_array) == 0 or not np.isfinite(idx_array).all():
        raise ValueError("Labels must be nonempty and finite")
    if not np.array_equal(np.unique(idx_array), np.arange(len(np.unique(idx_array)))):
        raise ValueError("Labels must be consecutive integers starting at zero")
    label_dict = {"num_class": len(set(idx_array)), "path2idx_dict": {}}
    for path, idx in zip(path_list, idx_array):
        key = get_dcase_path(path)
        if key in label_dict["path2idx_dict"]:
            raise ValueError(f"Duplicate training path: {key}")
        label_dict["path2idx_dict"][key] = int(idx)
    return label_dict


def write_pseudo_label_json(
    json_path: Path, idx_array: np.ndarray, path_list: List[str]
) -> None:
    pseudo_labels = generate_pseudo_labels(idx_array, path_list)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(pseudo_labels, f, indent=4)
