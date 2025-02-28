# Copyright 2024 Takuya Fujimura

import logging
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import hmean
from sklearn.metrics import roc_auc_score

from asdlib.utils.config_class import CustomHmeanConfig
from asdlib.utils.dcase_utils import get_dcase_idx

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
    auc_domain_list = [
        "s_auc",
        "t_auc",
        "smix_auc",
        "tmix_auc",
        "mix_auc",
        "s_pauc",
        "t_pauc",
        "mix_pauc",
    ]

    all_section = np.unique(score_df["section"].values)  # type: ignore

    auc_type_list = []
    for section in all_section:
        for auc_domain in auc_domain_list:
            auc_type_list.append(f"{section}_{auc_domain}")
    return auc_type_list


def get_official_hmean_cols(hmean_name: str, auc_type_list: List[str]) -> List[str]:
    section = sorted(list({int(auc_type.split("_")[0]) for auc_type in auc_type_list}))
    if hmean_name in ["official23", "official24"]:
        if section == [0]:
            return ["0_smix_auc", "0_tmix_auc", "0_mix_pauc"]
        else:
            logger.warning(f"{hmean_name}: section {section} is provided.")
            return []

    elif hmean_name.split("-")[0] in ["official22", "official21"]:
        if section != list(range(6)):
            logger.warning(f"{hmean_name}: section {section} is provided.")
            return []
        if hmean_name == "official22-dev":
            return [
                f"{i}_{a}"
                for i in range(3)
                for a in ["smix_auc", "tmix_auc", "mix_pauc"]
            ]
        elif hmean_name == "official22-eval":
            return [
                f"{i}_{a}"
                for i in range(3, 6)
                for a in ["smix_auc", "tmix_auc", "mix_pauc"]
            ]
        elif hmean_name == "official21-dev":
            return [
                f"{i}_{a}"
                for i in range(3)
                for a in ["s_auc", "t_auc", "s_pauc", "t_pauc"]
            ]
        elif hmean_name == "official21-eval":
            return [
                f"{i}_{a}"
                for i in range(3, 6)
                for a in ["s_auc", "t_auc", "s_pauc", "t_pauc"]
            ]
        else:
            raise NotImplementedError()

    else:
        raise NotImplementedError()


def add_hmean(
    evaluate_df: pd.DataFrame,
    hmean_cfg_list: List[str | CustomHmeanConfig],
    auc_type_list: List[str],
) -> pd.DataFrame:
    for hmean_cfg in hmean_cfg_list:
        if isinstance(hmean_cfg, str):
            hmean_name = hmean_cfg
            hmean_cols = get_official_hmean_cols(
                hmean_name=hmean_cfg, auc_type_list=auc_type_list
            )
        elif isinstance(hmean_cfg, CustomHmeanConfig):
            hmean_name = hmean_cfg.name
            hmean_cols = hmean_cfg.cols
        else:
            raise ValueError(f"Unexpected hmean_cfg type: {hmean_cfg}")

        if len(hmean_cols) > 0:
            evaluate_df[hmean_name] = evaluate_df[hmean_cols].apply(
                lambda x: hmean(x), axis=1
            )
    return evaluate_df
