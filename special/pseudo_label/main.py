import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import numpy as np
from models import PseudoLabelModel
from omegaconf import DictConfig
from pydantic import BaseModel
from tqdm import tqdm

from asdkit.datasets.collators import get_relative_dcase_path
from asdkit.utils.common import instantiate_tgt, write_json
from asdkit.utils.dcase_utils import MACHINE_DICT

logger = logging.getLogger(__name__)


class Config(BaseModel):
    dcase: str
    results_root_dir: Path
    labels_root_dir: Path
    recipe_dir: Path
    output_label_name: str
    machinewise: bool = True
    model: Dict[str, Any]


def generate_pseudo_labels(
    idx_array: np.ndarray, path_list: List[str]
) -> Dict[str, Any]:
    label_dict = {"num_class": len(set(idx_array)), "path2idx_dict": {}}
    for path, idx in zip(path_list, idx_array):
        label_dict["path2idx_dict"][get_relative_dcase_path(path)] = int(idx)
    return label_dict


def process_machinewise_clustering(
    all_machine_list: List[str], recipe_dir: Path, model_config: Dict[str, Any]
) -> Tuple[np.ndarray, List[str]]:
    """Perform clustering for each machine separately"""
    path_list: List[str] = []
    idx_array_list: List[np.ndarray] = []
    idx_offset: int = 0

    for machine in tqdm(all_machine_list):
        model: PseudoLabelModel = instantiate_tgt(model_config)
        with np.load(recipe_dir / machine / "train_extract.npz") as npz:
            embed = npz["embed"]
            path_list.extend(npz["path"].tolist())

        idx_array_tmp = model.fit_predict(embed=embed)
        idx_array_list.append(idx_array_tmp + idx_offset)
        idx_offset += len(set(idx_array_tmp))

    idx_array = np.concatenate(idx_array_list)
    return idx_array, path_list


def process_unified_clustering(
    all_machine_list: List[str], recipe_dir: Path, model_config: Dict[str, Any]
) -> Tuple[np.ndarray, List[str]]:
    """Perform clustering for all machines together"""
    model: PseudoLabelModel = instantiate_tgt(model_config)
    embed_list: List[np.ndarray] = []
    path_list: List[str] = []

    for machine in tqdm(all_machine_list):
        with np.load(recipe_dir / machine / "train_extract.npz") as npz:
            embed_list.append(npz["embed"])
            path_list.extend(npz["path"].tolist())

    idx_array = model.fit_predict(embed=np.vstack(embed_list))
    return idx_array, path_list


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(hydra_cfg: DictConfig) -> None:
    cfg: Config = Config(**hydra_cfg)

    json_path: Path = cfg.labels_root_dir / cfg.dcase / f"{cfg.output_label_name}.json"
    if json_path.exists():
        raise FileExistsError(f"{json_path} already exists.")

    all_machine_list: List[str] = list(
        set(MACHINE_DICT[f"{cfg.dcase}-dev"] + MACHINE_DICT[f"{cfg.dcase}-eval"])
    )
    all_machine_list.sort()
    recipe_dir: Path = cfg.results_root_dir / cfg.dcase / cfg.recipe_dir

    # Create mutable copy of model config
    model_config: Dict[str, Any] = dict(cfg.model)

    if cfg.machinewise:
        idx_array, path_list = process_machinewise_clustering(
            all_machine_list, recipe_dir, model_config
        )
    else:
        logger.info(
            "Note that the number of classes will be multiplied by the number of machines."
        )
        model_config["num_class"] *= len(all_machine_list)

        idx_array, path_list = process_unified_clustering(
            all_machine_list, recipe_dir, model_config
        )

    pseudo_labels: Dict[str, Any] = generate_pseudo_labels(idx_array, path_list)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(json_path=json_path, data=pseudo_labels)


if __name__ == "__main__":
    main()
