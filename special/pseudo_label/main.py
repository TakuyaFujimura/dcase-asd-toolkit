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
from tqdm import tqdm

from asdkit.utils.common import instantiate_tgt

logger = logging.getLogger(__name__)


class Config(BaseModel):
    dcase: str
    results_root_dir: Path
    labels_root_dir: Path
    teacher_dir: Path
    output_label_name: str
    machinewise: bool = True
    model: Dict[str, Any]


def process_machinewise_clustering(
    all_machine_list: List[str], teacher_dir: Path, model_config: Dict[str, Any]
) -> Tuple[np.ndarray, List[str]]:
    """Perform clustering for each machine separately"""
    path_list: List[str] = []
    idx_array_list: List[np.ndarray] = []
    idx_offset: int = 0

    for machine in tqdm(all_machine_list):
        model: PseudoLabelModel = instantiate_tgt(model_config)
        extract = load_extract(teacher_dir=teacher_dir, machine=machine)
        embed = extract.embed
        path = extract.path
        path_list.extend(path)

        idx_array_tmp = model.fit_predict(embed=embed, path=path)
        idx_array_list.append(idx_array_tmp + idx_offset)
        idx_offset += len(set(idx_array_tmp))

    idx_array = np.concatenate(idx_array_list)
    return idx_array, path_list


def process_unified_clustering(
    all_machine_list: List[str], teacher_dir: Path, model_config: Dict[str, Any]
) -> Tuple[np.ndarray, List[str]]:
    """Perform clustering for all machines together"""
    model: PseudoLabelModel = instantiate_tgt(model_config)
    embed_list: List[np.ndarray] = []
    path_list: List[str] = []

    for machine in tqdm(all_machine_list):
        extract = load_extract(teacher_dir=teacher_dir, machine=machine)
        embed_list.append(extract.embed)
        path_list.extend(extract.path)

    idx_array = model.fit_predict(embed=np.vstack(embed_list), path=path_list)
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

    # Create mutable copy of model config
    model_config: Dict[str, Any] = dict(cfg.model)

    if cfg.machinewise:
        idx_array, path_list = process_machinewise_clustering(
            all_machine_list, teacher_dir, model_config
        )
    else:
        if "num_class" in model_config:
            model_config["num_class"] *= len(all_machine_list)
            logger.info(
                "Note that the number of classes will be multiplied by the number of machines."
            )
        else:
            logger.info("ratio-based clustering will be performed")
            assert "ratio" in model_config

        idx_array, path_list = process_unified_clustering(
            all_machine_list, teacher_dir, model_config
        )

    write_pseudo_label_json(
        json_path=json_path, idx_array=idx_array, path_list=path_list
    )


if __name__ == "__main__":
    main()
