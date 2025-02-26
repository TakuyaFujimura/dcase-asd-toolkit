# Copyright 2024 Takuya Fujimura
import copy
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
from hydra.utils import instantiate

from ..backends import *
from ..utils.config_class.main_test_config import MainTestConfig

logger = logging.getLogger(__name__)


def get_name(backend_cfg: Dict[str, Any]) -> str:
    join_list = [backend_cfg["_target_"].split(".")[-1]]
    join_list += [str(v) for v in backend_cfg["hp_dict"].values()]
    return "-".join(join_list)


def rm_unnecesary_col(df: pd.DataFrame) -> pd.DataFrame:
    rm_cols = []
    for col in df.keys():
        if col.startswith("e_") and int(col.split("_")[-1]) >= 0:
            rm_cols.append(col)
        elif col.startswith("l_") and int(col.split("_")[-1]) >= 0:
            rm_cols.append(col)

    return df.drop(rm_cols, axis=1)


def get_dicts(
    machine_dir: Path,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    org_df_dict: Dict[str, pd.DataFrame] = {}
    output_df_dict: Dict[str, pd.DataFrame] = {}
    for split in ["train", "test"]:
        org_df_dict[split] = pd.read_csv(machine_dir / f"{split}_extraction.csv")
        output_df_dict[split] = copy.deepcopy(org_df_dict[split])
        output_df_dict[split] = rm_unnecesary_col(output_df_dict[split])
    return org_df_dict, output_df_dict


def add_score(
    backend_cfg: Dict[str, Any],
    org_df_dict: Dict[str, pd.DataFrame],
    output_df_dict: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    backend = instantiate(backend_cfg)
    backend.fit(org_df_dict["train"])
    backend_name = get_name(backend_cfg)
    for split in ["train", "test"]:
        anomaly_score_dict = backend.anomaly_score(org_df_dict[split])
        for key, score in anomaly_score_dict.items():
            output_df_dict[split][f"AS-{backend_name}-{key}"] = score
    return output_df_dict


def score_main(cfg: MainTestConfig, machine_dir: Path) -> None:

    logger.info(f"Start scoring {machine_dir}")

    org_df_dict, output_df_dict = get_dicts(machine_dir=machine_dir)

    # Loop for backend
    for backend_cfg in cfg.backend:
        output_df_dict = add_score(
            backend_cfg=backend_cfg,
            org_df_dict=org_df_dict,
            output_df_dict=output_df_dict,
        )

    # Save
    for split, output_df in output_df_dict.items():
        output_path = machine_dir / f"{split}_score.csv"
        output_df.to_csv(output_path, index=False)
        logging.info(f"Saved at {str(output_path)}")
