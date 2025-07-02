from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from asdkit.backends.base import BaseBackend
from asdkit.utils.common.instantiate_util import instantiate_tgt
from asdkit.utils.common.match import re_match_any


def get_extract_score_dicts(
    output_dir: Path, extract_items: List[str]
) -> Tuple[Dict[str, dict], Dict[str, pd.DataFrame]]:
    extract_dict_dict: Dict[str, dict] = {}
    score_df_dict: Dict[str, pd.DataFrame] = {}

    for split in ["train", "test"]:
        extract_dict = {}
        score_dict = {}
        with np.load(output_dir / f"{split}_extract.npz") as npz:
            for key, value in npz.items():
                extract_dict[key] = value
                if re_match_any(patterns=extract_items, string=key):
                    score_dict[key] = value

        extract_dict_dict[split] = extract_dict
        score_df_dict[split] = pd.DataFrame(score_dict)
    return extract_dict_dict, score_df_dict


def get_as_name(backend_cfg: Dict[str, Any]) -> str:
    join_list = [backend_cfg["tgt_class"].split(".")[-1]]
    join_list += [str(v) for k, v in backend_cfg.items() if k != "tgt_class"]
    return "-".join(join_list)


def add_score(
    backend_cfg: Dict[str, Any],
    extract_dict_dict: Dict[str, dict],
    score_df_dict: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    backend: BaseBackend = instantiate_tgt(backend_cfg)
    backend.fit(extract_dict_dict["train"])
    backend_name = get_as_name(backend_cfg)
    for split in ["train", "test"]:
        anomaly_score_dict = backend.anomaly_score(extract_dict_dict[split])
        for key, score in anomaly_score_dict.items():
            score_df_dict[split][f"AS-{backend_name}-{key}"] = score
    return score_df_dict
