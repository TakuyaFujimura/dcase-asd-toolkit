# Copyright 2024 Takuya Fujimura

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from asdkit.utils.dcase_utils import MACHINE_DICT
from asdkit.utils.dcase_utils.evaluation_rules import EVALUATION_RULES, average_metrics

logger = logging.getLogger(__name__)


def resolve_table_metric(dcase: str, subset: str, metric: str) -> Optional[str]:
    """Resolve a requested metric to the column used by a DCASE edition."""
    if dcase not in EVALUATION_RULES:
        raise ValueError(f"Unknown dcase: {dcase}")
    rules = EVALUATION_RULES[dcase]
    if not rules.supports_domains:
        # No domain-specific metrics
        if metric in ["t_auc", "t_pauc", "smix_auc", "tmix_auc", "mix_auc", "mix_pauc"]:
            return None
    prefix = subset if rules.multiple_sections else "0"
    return f"{prefix}_{metric}"


def build_metric_table(
    output_dir: Path, dcase: str, subset: str, metric: str
) -> Optional[pd.DataFrame]:
    # Initialize
    machine_frames = []
    machines = MACHINE_DICT[f"{dcase}-{subset}"]
    metric_column = resolve_table_metric(dcase=dcase, subset=subset, metric=metric)
    if metric_column is None:
        return None

    # Load metrics for each machine
    for i, machine in enumerate(machines):
        evaluate_df = pd.read_csv(output_dir / machine / "test_evaluate.csv")
        if metric_column not in evaluate_df:
            logger.warning(
                f"Metric {metric_column} not found in {output_dir / machine / 'test_evaluate.csv'}. This metric will be skipped."
            )
            return None

        # Keep common backends
        if i == 0:
            backends = evaluate_df.backend.values
        elif set(backends) != set(evaluate_df.backend.values):
            backends = np.intersect1d(backends, evaluate_df.backend.values)
            logger.warning(
                "Different backends found across machines. Using common backends."
            )
        machine_frames.append(evaluate_df)

    # Align backend rows
    backends = np.sort(backends)
    metric_columns = []
    for df in machine_frames:
        ordered_df = df[df["backend"].isin(backends)]
        ordered_df = ordered_df.set_index("backend").loc[backends].reset_index()
        metric_columns.append(ordered_df[metric_column])
    table_df = pd.concat(metric_columns, axis=1)
    table_df.columns = machines

    # Calculate arithmetic or harmonic mean
    table_df["total"] = average_metrics(values=table_df, rules=EVALUATION_RULES[dcase])

    table_df.index = backends  # type: ignore
    table_df = table_df.reset_index().rename(columns={"index": "backend"})
    return table_df
