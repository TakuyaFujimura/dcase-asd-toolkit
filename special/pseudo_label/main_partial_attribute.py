import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import numpy as np
from common import (
    get_all_machine_list,
    get_output_json_path,
    get_teacher_dir,
    load_extract,
    write_pseudo_label_json,
)
from models import PseudoLabelModel
from omegaconf import DictConfig
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from asdkit.utils.common import instantiate_tgt, read_json
from asdkit.utils.dcase_utils import MACHINE_DICT

logger = logging.getLogger(__name__)


class Config(BaseModel):
    dcase: str
    results_root_dir: Path
    labels_root_dir: Path
    teacher_dir: Path
    output_label_name: str
    model: Dict[str, Any]
    org_label_name: str


def get_org_label(
    org_path2idx_dict: Dict[str, int], machine: str
) -> Tuple[np.ndarray, List[str]]:
    path_list: List[str] = []
    idx_list: List[int] = []
    for path, idx in org_path2idx_dict.items():
        split_path = path.split("/")
        if split_path[-2] == "train" and split_path[-3] == machine:
            path_list.append(path)
            idx_list.append(idx)
    idx_array = np.array(idx_list)
    return idx_array, path_list


def get_pseudo_label(
    teacher_dir: Path, machine: str, model_config: Dict[str, Any]
) -> Tuple[np.ndarray, List[str]]:
    model: PseudoLabelModel = instantiate_tgt(model_config)
    extract = load_extract(teacher_dir=teacher_dir, machine=machine)
    embed = extract.embed
    path = extract.path

    idx_array = model.fit_predict(embed=embed, path=path)
    return idx_array, path


def combine_idx_array_list(idx_array_list: List[np.ndarray]) -> np.ndarray:
    idx_array = np.array([])
    for i, idx_array_tmp in enumerate(idx_array_list):
        idx_array_tmp = [f"{i}_{idx}" for idx in idx_array_tmp]
        idx_array = np.concatenate((idx_array, np.array(idx_array_tmp)))
    idx_array = LabelEncoder().fit_transform(idx_array)
    return idx_array


def process_machinewise_labels(
    all_machine_list: List[str],
    teacher_dir: Path,
    model_config: Dict[str, Any],
    org_path2idx_dict: Dict[str, int],
    wo_attr_machine_list: List[str],
) -> Tuple[np.ndarray, List[str]]:
    idx_array_list: List[np.ndarray] = []
    path_list: List[str] = []
    wo_attr_machine_set = set(wo_attr_machine_list)

    for machine in tqdm(all_machine_list):
        if machine in wo_attr_machine_set:
            idx_array, path_list_tmp = get_pseudo_label(
                teacher_dir=teacher_dir, machine=machine, model_config=model_config
            )
            logger.info(f"{machine} no attr: {len(np.unique(idx_array))} clusters")
        else:
            idx_array, path_list_tmp = get_org_label(
                org_path2idx_dict=org_path2idx_dict, machine=machine
            )
            logger.info(f"{machine} org attr: {len(np.unique(idx_array))} clusters")
        idx_array_list.append(idx_array)
        path_list.extend(path_list_tmp)

    assert len(path_list) == len(set(path_list))
    idx_array = combine_idx_array_list(idx_array_list=idx_array_list)
    return idx_array, path_list


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(hydra_cfg: DictConfig) -> None:
    cfg: Config = Config(**hydra_cfg)

    json_path = get_output_json_path(
        labels_root_dir=cfg.labels_root_dir,
        dcase=cfg.dcase,
        output_label_name=cfg.output_label_name,
    )
    all_machine_list = get_all_machine_list(dcase=cfg.dcase)
    teacher_dir = get_teacher_dir(
        results_root_dir=cfg.results_root_dir,
        dcase=cfg.dcase,
        teacher_dir=cfg.teacher_dir,
    )
    model_config: Dict[str, Any] = dict(cfg.model)
    org_path2idx_dict: Dict[str, int] = read_json(
        f"../../labels/{cfg.dcase}/{cfg.org_label_name}.json"
    )["path2idx_dict"]  # type: ignore
    wo_attr_machine_list: List[str] = MACHINE_DICT[f"{cfg.dcase}-wo-attr"]

    idx_array, path_list = process_machinewise_labels(
        all_machine_list=all_machine_list,
        teacher_dir=teacher_dir,
        model_config=model_config,
        org_path2idx_dict=org_path2idx_dict,
        wo_attr_machine_list=wo_attr_machine_list,
    )

    write_pseudo_label_json(
        json_path=json_path, idx_array=idx_array, path_list=path_list
    )


if __name__ == "__main__":
    main()
