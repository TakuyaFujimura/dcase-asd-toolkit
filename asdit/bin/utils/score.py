from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from asdit.utils.common.df_util import pickup_cols
from asdit.utils.common.instantiate_util import instantiate_tgt
from asdit.utils.dcase_utils import INFOLIST


def get_dicts(
    output_dir: Path, extract_items: List[str]
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    extract_df_dict: Dict[str, pd.DataFrame] = {}
    score_df_dict: Dict[str, pd.DataFrame] = {}
    extract_items = list(set(INFOLIST + extract_items))
    for split in ["train", "test"]:
        extract_df_dict[split] = pd.read_csv(output_dir / f"{split}_extract.csv")
        score_df_dict[split] = pickup_cols(
            df=extract_df_dict[split], extract_items=extract_items
        )
    return extract_df_dict, score_df_dict


def get_as_name(backend_cfg: Dict[str, Any]) -> str:
    join_list = [backend_cfg["tgt_class"].split(".")[-1]]
    join_list += [str(v) for k, v in backend_cfg.items() if k != "tgt_class"]
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
