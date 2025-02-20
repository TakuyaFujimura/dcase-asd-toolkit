# Copyright 2024 Takuya Fujimura

import logging
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import hmean
from sklearn.metrics import roc_auc_score

from ...utils.config_class.main_test_config import EvaluateConfig

logger = logging.getLogger(__name__)


def get_score_name(score_df: pd.DataFrame):
    score_name_list = [col for col in score_df.columns if col.split("-")[0] == "AS"]
    return score_name_list


def get_idx(
    auc_type: str, section: np.ndarray, is_target: np.ndarray, is_normal: np.ndarray
) -> np.ndarray:
    domain = auc_type.split("_")[1]
    if domain == "s":
        domain_idx = is_target == 0
    elif auc_type == "t":
        domain_idx = is_target == 1
    elif domain == "smix":
        domain_idx = (is_target == 0) | (is_normal == 0)
    elif domain == "tmix":
        domain_idx = (is_target == 1) | (is_normal == 0)
    elif domain == "mix":
        domain_idx = np.ones_like(is_normal).astype(bool)
    else:
        raise NotImplementedError()

    section_idx = section == int(auc_type.split("_")[0])
    return domain_idx & section_idx


def dcase_auc(
    anomaly_score: np.ndarray,
    section: np.ndarray,
    is_normal: np.ndarray,
    is_target: np.ndarray,
    auc_type: str,
) -> float:
    is_anomaly = 1 - is_normal

    auc_pauc = auc_type.split("_")[2]

    idx = get_idx(
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


def add_hmean(result_df: pd.DataFrame, hmean_type: str) -> pd.DataFrame:
    # examples of hmean_type are "official22-dev", "official23", ...
    hmean_cols = []

    if hmean_type in ["official23", "official24"]:
        hmean_cols = ["0_smix_auc", "0_tmix_auc", "0_mix_pauc"]
    elif hmean_type == "official22-dev":
        for i in range(3):
            hmean_cols += [f"{i}_smix_auc", f"{i}_tmix_auc", f"{i}_mix_pauc"]
    elif hmean_type == "official22-eval":
        for i in range(3, 6):
            hmean_cols += [f"{i}_smix_auc", f"{i}_tmix_auc", f"{i}_mix_pauc"]
    elif hmean_type == "official21-dev":
        for i in range(3):
            hmean_cols += [f"{i}_s_auc", f"{i}_t_auc", f"{i}_s_pauc", f"{i}_t_pauc"]
    elif hmean_type == "official21-eval":
        for i in range(3, 6):
            hmean_cols += [f"{i}_s_auc", f"{i}_t_auc", f"{i}_s_pauc", f"{i}_t_pauc"]
    else:
        raise NotImplementedError()

    result_df[hmean_type] = result_df[hmean_cols].apply(lambda x: hmean(x), axis=1)
    return result_df


def evaluate(evaluate_cfg: EvaluateConfig, score_df: pd.DataFrame) -> pd.DataFrame:
    score_name_list = get_score_name(score_df)
    auc_type_list = get_auc_type_list(score_df)
    result_df = pd.DataFrame(index=score_name_list, columns=auc_type_list)
    for score_name in score_name_list:
        for auc_type in auc_type_list:
            result_df.at[score_name, auc_type] = dcase_auc(
                anomaly_score=score_df[score_name].values,  # type: ignore
                section=score_df["section"].values,  # type: ignore
                is_normal=score_df["is_normal"].values,  # type: ignore
                is_target=score_df["is_target"].values,  # type: ignore
                auc_type=auc_type,
            )
    for hmean_type in evaluate_cfg.hmean_list:
        result_df = add_hmean(result_df, hmean_type)

    return result_df
