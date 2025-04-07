# Copyright 2024 Takuya Fujimura

import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import hmean
from sklearn.metrics import roc_auc_score

from asdit.utils.dcase_utils import get_dcase_idx

logger = logging.getLogger(__name__)


def get_as_name(score_df: pd.DataFrame):
    as_name_list = [col for col in score_df.columns if col.split("-")[0] == "AS"]
    return as_name_list


def dcase_auc(
    anomaly_score: np.ndarray,
    section: np.ndarray,
    is_normal: np.ndarray,
    is_target: np.ndarray,
    auc_type: str,
) -> float:
    is_anomaly = 1 - is_normal

    auc_pauc = auc_type.split("_")[2]

    idx = get_dcase_idx(
        auc_type=auc_type, section=section, is_target=is_target, is_normal=is_normal
    )

    if auc_pauc == "auc":
        auc_score = roc_auc_score(y_true=is_anomaly[idx], y_score=anomaly_score[idx])
    elif auc_pauc == "pauc":
        auc_score = roc_auc_score(
            y_true=is_anomaly[idx], y_score=anomaly_score[idx], max_fpr=0.1
        )
    else:
        raise NotImplementedError()
    return auc_score  # type: ignore


def get_auc_type_list(score_df: pd.DataFrame) -> List[str]:
    auc_domain_list = []

    all_section = np.unique(score_df["section"].values)  # type: ignore
    all_is_target = np.unique(score_df["is_target"].values)  # type: ignore

    if 0 in all_is_target:
        auc_domain_list += ["s_auc", "s_pauc"]
    if 1 in all_is_target:
        auc_domain_list += ["t_auc", "t_pauc"]
    if 0 in all_is_target and 1 in all_is_target:
        auc_domain_list += ["smix_auc", "tmix_auc", "mix_auc", "mix_pauc"]

    auc_type_list = []
    for section in all_section:
        for auc_domain in auc_domain_list:
            auc_type_list.append(f"{section}_{auc_domain}")
    return auc_type_list


def get_official_metriclist(dcase: str) -> List[str]:
    if dcase == "dcase2020":
        return ["s_auc", "s_pauc"]
    elif dcase == "dcase2021":
        return ["s_auc", "t_auc", "s_pauc", "t_pauc"]
    elif dcase in ["dcase2022", "dcase2023", "dcase2024"]:
        return ["smix_auc", "tmix_auc", "mix_pauc"]
    else:
        raise NotImplementedError()


def get_official_sectionlist(
    dcase: str, split: str = "", machine: str = ""
) -> List[int]:

    if dcase == "dcase2020":
        section_dict = {
            "fan": {"dev": [0, 2, 4, 6], "eval": [1, 3, 5]},
            "pump": {"dev": [0, 2, 4, 6], "eval": [1, 3, 5]},
            "slider": {"dev": [0, 2, 4, 6], "eval": [1, 3, 5]},
            "ToyCar": {"dev": [1, 2, 3, 4], "eval": [5, 6, 7]},
            "ToyConveyor": {"dev": [1, 2, 3], "eval": [4, 5, 6]},
            "valve": {"dev": [0, 2, 4, 6], "eval": [1, 3, 5]},
        }
        return section_dict[machine][split]
    elif dcase in ["dcase2021", "dcase2022"]:
        if split == "dev":
            return [0, 1, 2]
        elif split == "eval":
            return [3, 4, 5]
        else:
            raise NotImplementedError()
    elif dcase in ["dcase2023", "dcase2024"]:
        return [0]
    else:
        raise NotImplementedError()


def combine_section_metric(sectionlist: List[int], metriclist: List[str]) -> List[str]:
    return [f"{section}_{metric}" for section in sectionlist for metric in metriclist]


def complete_hmean_cfg(
    hmean_cfg_dict: Dict[str, List[str]], dcase: str, machine: str
) -> Dict[str, List[str]]:
    hmean_cfg_dict_new = {}
    for hmean_name, metriclist in hmean_cfg_dict.items():
        if dcase in ["dcase2020", "dcase2021", "dcase2022"]:
            for split in ["dev", "eval"]:
                hmean_cfg_dict_new[f"{hmean_name}-{split}"] = combine_section_metric(
                    sectionlist=get_official_sectionlist(
                        dcase=dcase, split=split, machine=machine
                    ),
                    metriclist=metriclist,
                )
        elif dcase in ["dcase2023", "dcase2024"]:
            hmean_cfg_dict_new[hmean_name] = combine_section_metric(
                sectionlist=get_official_sectionlist(dcase=dcase), metriclist=metriclist
            )
        else:
            raise NotImplementedError()

    return hmean_cfg_dict_new


def hmean_is_available(
    evaluate_df: pd.DataFrame, hmean_name: str, hmean_cols: List[str]
) -> bool:
    if len(hmean_cols) == 0:
        return False
    if not set(hmean_cols).issubset(evaluate_df.columns):
        return False
    return True


def add_hmean(
    evaluate_df: pd.DataFrame,
    dcase: str,
    hmean_cfg_dict: Dict[str, List[str]],
    machine: str,
) -> pd.DataFrame:

    # added official metriclist
    hmean_cfg_dict[f"official{dcase[-2:]}"] = get_official_metriclist(dcase=dcase)
    # added section to metric
    hmean_cfg_dict = complete_hmean_cfg(
        hmean_cfg_dict=hmean_cfg_dict, dcase=dcase, machine=machine
    )

    for hmean_name, hmean_cols in hmean_cfg_dict.items():
        if not hmean_is_available(
            evaluate_df=evaluate_df, hmean_name=hmean_name, hmean_cols=hmean_cols
        ):
            logger.warning(
                f"Skipped {hmean_name} because {hmean_cols} is not available."
            )

        evaluate_df[hmean_name] = evaluate_df[hmean_cols].apply(
            lambda x: hmean(x), axis=1
        )
    return evaluate_df
