import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


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
    teacher_dir: Path, machine: str, filename: str = "train_extract.npz"
) -> ExtractData:
    with np.load(teacher_dir / machine / filename) as npz:
        return ExtractData(embed=npz["embed"], path=npz["path"].tolist())


def load_embed(teacher_dir: Path, machine: str, filename: str) -> np.ndarray:
    with np.load(teacher_dir / machine / filename) as npz:
        return npz["embed"]


def _get_relative_dcase_path(path: str) -> str:
    path_split = path.split("/")
    assert path_split[-2] in ["train", "test", "supplemental"]
    assert path_split[-4] == "raw"
    assert path_split[-5].startswith("dcase202")
    return "/".join(path_split[-5:])


def generate_pseudo_labels(
    idx_array: np.ndarray, path_list: List[str]
) -> Dict[str, Any]:
    label_dict = {"num_class": len(set(idx_array)), "path2idx_dict": {}}
    for path, idx in zip(path_list, idx_array):
        label_dict["path2idx_dict"][_get_relative_dcase_path(path)] = int(idx)
    return label_dict


def write_pseudo_label_json(
    json_path: Path, idx_array: np.ndarray, path_list: List[str]
) -> None:
    pseudo_labels = generate_pseudo_labels(idx_array, path_list)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(pseudo_labels, f, indent=4)
