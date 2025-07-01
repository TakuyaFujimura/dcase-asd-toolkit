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


def get_auc_type_list_domain_auc(score_df: pd.DataFrame):
    domain_auc_list = []
    all_is_target = np.unique(score_df["is_target"].values)  # type: ignore

    if 0 in all_is_target:
        domain_auc_list += ["s_auc", "s_pauc"]
    if 1 in all_is_target:
        domain_auc_list += ["t_auc", "t_pauc"]
    if 0 in all_is_target and 1 in all_is_target:
        domain_auc_list += ["smix_auc", "tmix_auc", "mix_auc", "mix_pauc"]
    return domain_auc_list


def get_auc_type_list(score_df: pd.DataFrame) -> List[str]:
    all_section = np.unique(score_df["section"].values)  # type: ignore
    domain_auc_list = get_auc_type_list_domain_auc(score_df)

    auc_type_list = []
    for section in all_section:
        for domain_auc in domain_auc_list:
            auc_type_list.append(f"{int(section)}_{domain_auc}")
    return auc_type_list


# ----------------------------------------------------------------------- #


def get_official_domain_auc_list(dcase: str) -> List[str]:
    if dcase == "dcase2020":
        return ["s_auc", "s_pauc"]
    elif dcase == "dcase2021":
        return ["s_auc", "t_auc", "s_pauc", "t_pauc"]
    elif dcase in ["dcase2022", "dcase2023", "dcase2024"]:
        return ["smix_auc", "tmix_auc", "mix_pauc"]
    else:
        raise NotImplementedError()


def get_official_section_list(
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


def mix_auc_and_section(
    evaluate_df: pd.DataFrame,
    dcase: str,
    metric_name: str,
    sub_auc_list: List[str],
    sub_section_list: List[int],
) -> None:
    sub_metric_list = [
        f"{section}_{auc}" for section in sub_section_list for auc in sub_auc_list
    ]
    if len(sub_metric_list) == 0:
        logger.warning(f"Skipped {metric_name} because sub_metric_list is empty.")
        return
    if not set(sub_metric_list).issubset(evaluate_df.columns):
        logger.warning(
            f"Skipped {metric_name} because sub_metric_list {sub_metric_list} is not available in evaluate_df."
        )
        return

    if dcase == "dcase2020":
        evaluate_df[metric_name] = evaluate_df[sub_metric_list].apply(np.mean, axis=1)
    else:
        evaluate_df[metric_name] = evaluate_df[sub_metric_list].apply(
            lambda x: hmean(x), axis=1
        )


def add_official(
    evaluate_df: pd.DataFrame,
    dcase: str,
    machine: str,
) -> pd.DataFrame:

    sub_auc_list = get_official_domain_auc_list(dcase)

    if dcase in ["dcase2020", "dcase2021", "dcase2022"]:
        for split in ["dev", "eval"]:
            metric_name = f"{split}_official{dcase[-2:]}"
            sub_section_list = get_official_section_list(
                dcase=dcase, split=split, machine=machine
            )
            mix_auc_and_section(
                evaluate_df=evaluate_df,
                dcase=dcase,
                metric_name=metric_name,
                sub_auc_list=sub_auc_list,
                sub_section_list=sub_section_list,
            )
    elif dcase in ["dcase2023", "dcase2024"]:
        metric_name = f"0_official{dcase[-2:]}"
        sub_section_list = get_official_section_list(dcase=dcase)
        mix_auc_and_section(
            evaluate_df=evaluate_df,
            dcase=dcase,
            metric_name=metric_name,
            sub_auc_list=sub_auc_list,
            sub_section_list=sub_section_list,
        )
    else:
        raise NotImplementedError()

    return evaluate_df


def add_split_total(
    evaluate_df: pd.DataFrame,
    dcase: str,
    machine: str,
    domain_auc: str,
) -> pd.DataFrame:
    sub_auc_list = [domain_auc]
    if dcase in ["dcase2020", "dcase2021", "dcase2022"]:
        for split in ["dev", "eval"]:
            metric_name = f"{split}_{domain_auc}"
            sub_section_list = get_official_section_list(
                dcase=dcase, split=split, machine=machine
            )
            mix_auc_and_section(
                evaluate_df=evaluate_df,
                dcase=dcase,
                metric_name=metric_name,
                sub_auc_list=sub_auc_list,
                sub_section_list=sub_section_list,
            )
    elif dcase in ["dcase2023", "dcase2024"]:
        # Only one section
        return evaluate_df
    else:
        raise NotImplementedError()

    return evaluate_df
