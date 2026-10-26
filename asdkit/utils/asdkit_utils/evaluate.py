# Copyright 2024 Takuya Fujimura

import logging
from typing import Dict, List

import numpy as np
import pandas as pd

from asdkit.utils.dcase_utils.evaluation import dcase_auc
from asdkit.utils.dcase_utils.evaluation_rules import EVALUATION_RULES, average_metrics

logger = logging.getLogger(__name__)


def get_anomaly_score_columns(score_df: pd.DataFrame) -> List[str]:
    return [column for column in score_df.columns if column.split("-")[0] == "AS"]


def get_domain_metrics(score_df: pd.DataFrame) -> List[str]:
    domain_metrics = []
    domains = np.unique(score_df["is_target"].values)  # type: ignore

    if 0 in domains:
        domain_metrics.extend(["s_auc", "s_pauc"])
    if 1 in domains:
        domain_metrics.extend(["t_auc", "t_pauc"])
    if 0 in domains and 1 in domains:
        domain_metrics.extend(["smix_auc", "tmix_auc", "mix_auc", "mix_pauc"])
    return domain_metrics


def append_aggregate_metric(
    evaluate_df: pd.DataFrame,
    dcase: str,
    metric_name: str,
    domain_metrics: List[str],
    sections: List[int],
) -> None:
    """Append one mean column in place; skip incomplete groups with a warning."""
    columns = [
        f"{section_id}_{domain_metric}"
        for section_id in sections
        for domain_metric in domain_metrics
    ]
    if len(columns) == 0:
        logger.warning(f"Skipped {metric_name} because columns is empty.")
        return
    if not set(columns).issubset(evaluate_df.columns):
        logger.warning(
            f"Skipped {metric_name} because columns {columns} is not available in evaluate_df."
        )
        return

    evaluate_df[metric_name] = average_metrics(
        values=evaluate_df[columns], rules=EVALUATION_RULES[dcase]
    )


def evaluate_scores(score_df: pd.DataFrame, dcase: str, machine: str) -> pd.DataFrame:
    """Compute section, subset, and official metrics in their CSV column order."""
    score_columns = get_anomaly_score_columns(score_df)
    section_ids = score_df["section"].values
    unique_section_ids = np.unique(section_ids)  # type: ignore
    domain_metrics = get_domain_metrics(score_df)
    evaluate_df = pd.DataFrame(index=score_columns)

    # Compute metrics for each anomaly score column, section, and domain
    for score_column in score_columns:
        for section_id in unique_section_ids:
            for domain_metric in domain_metrics:
                auc_type = f"{int(section_id)}_{domain_metric}"
                evaluate_df.at[score_column, auc_type] = dcase_auc(
                    anomaly_score=score_df[score_column].values,  # type: ignore
                    section_ids=section_ids,  # type: ignore
                    is_normal=score_df["is_normal"].values,  # type: ignore
                    is_target=score_df["is_target"].values,  # type: ignore
                    auc_type=auc_type,
                )

    # Prepare section groups
    rules = EVALUATION_RULES[dcase]
    if not rules.multiple_sections:
        section_groups = {"0": [0]}
    elif rules.sections_by_machine_and_subset is not None:
        section_groups = rules.sections_by_machine_and_subset[machine]
    else:
        assert rules.sections_by_subset is not None
        section_groups = rules.sections_by_subset

    # Compute aggregate metrics for each section group and domain metric
    if rules.multiple_sections:
        for metric in domain_metrics:
            for prefix, sections in section_groups.items():
                append_aggregate_metric(
                    evaluate_df=evaluate_df,
                    dcase=dcase,
                    metric_name=f"{prefix}_{metric}",
                    domain_metrics=[metric],
                    sections=sections,
                )

    # Compute official totals by combining metrics and sections
    for prefix, sections in section_groups.items():
        append_aggregate_metric(
            evaluate_df=evaluate_df,
            dcase=dcase,
            metric_name=f"{prefix}_official{dcase[-2:]}",
            domain_metrics=list(rules.official_metrics),
            sections=sections,
        )

    evaluate_df = evaluate_df.reset_index().rename(columns={"index": "backend"})
    return evaluate_df
