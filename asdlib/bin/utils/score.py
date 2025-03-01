import copy
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

from asdlib.utils.common.instantiate_util import instantiate_tgt


def rm_unnecesary_col(df: pd.DataFrame) -> pd.DataFrame:
    rm_cols = []
    for col in df.keys():
        if col.startswith("e_") and int(col.split("_")[-1]) >= 0:
            rm_cols.append(col)
        elif col.startswith("l_") and int(col.split("_")[-1]) >= 0:
            rm_cols.append(col)
    return df.drop(rm_cols, axis=1)


def get_dicts(
    output_dir: Path,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    extract_df_dict: Dict[str, pd.DataFrame] = {}
    score_df_dict: Dict[str, pd.DataFrame] = {}
    for split in ["train", "test"]:
        extract_df_dict[split] = pd.read_csv(output_dir / f"{split}_extract.csv")
        score_df_dict[split] = copy.deepcopy(extract_df_dict[split])
        score_df_dict[split] = rm_unnecesary_col(score_df_dict[split])
    return extract_df_dict, score_df_dict


def get_as_name(backend_cfg: Dict[str, Any]) -> str:
    join_list = [backend_cfg["tgt_class"].split(".")[-1]]
    join_list += [str(v) for v in backend_cfg["hp_dict"].values()]
    return "-".join(join_list)


def add_score(
    backend_cfg: Dict[str, Any],
    extract_df_dict: Dict[str, pd.DataFrame],
    score_df_dict: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    backend = instantiate_tgt(backend_cfg)
    backend.fit(extract_df_dict["train"])
    backend_name = get_as_name(backend_cfg)
    for split in ["train", "test"]:
        anomaly_score_dict = backend.anomaly_score(extract_df_dict[split])
        for key, score in anomaly_score_dict.items():
            score_df_dict[split][f"AS-{backend_name}-{key}"] = score
    return score_df_dict
