"""Edition-dependent conventions shared by evaluation and table generation."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import hmean


@dataclass(frozen=True)
class EvaluationRules:
    official_metrics: Tuple[str, ...]
    multiple_sections: bool
    aggregation_method: str = "harmonic"
    supports_domains: bool = True
    sections_by_subset: Optional[Dict[str, List[int]]] = None
    sections_by_machine_and_subset: Optional[Dict[str, Dict[str, List[int]]]] = None


EVALUATION_RULES: Dict[str, EvaluationRules] = {
    "dcase2020": EvaluationRules(
        official_metrics=("s_auc", "s_pauc"),
        multiple_sections=True,
        aggregation_method="arithmetic",
        supports_domains=False,
        sections_by_machine_and_subset={
            "fan": {"dev": [0, 2, 4, 6], "eval": [1, 3, 5]},
            "pump": {"dev": [0, 2, 4, 6], "eval": [1, 3, 5]},
            "slider": {"dev": [0, 2, 4, 6], "eval": [1, 3, 5]},
            "ToyCar": {"dev": [1, 2, 3, 4], "eval": [5, 6, 7]},
            "ToyConveyor": {"dev": [1, 2, 3], "eval": [4, 5, 6]},
            "valve": {"dev": [0, 2, 4, 6], "eval": [1, 3, 5]},
        },
    ),
    "dcase2021": EvaluationRules(
        official_metrics=("s_auc", "t_auc", "s_pauc", "t_pauc"),
        multiple_sections=True,
        sections_by_subset={"dev": [0, 1, 2], "eval": [3, 4, 5]},
    ),
    "dcase2022": EvaluationRules(
        official_metrics=("smix_auc", "tmix_auc", "mix_pauc"),
        multiple_sections=True,
        sections_by_subset={"dev": [0, 1, 2], "eval": [3, 4, 5]},
    ),
    **{
        f"dcase{year}": EvaluationRules(
            official_metrics=("smix_auc", "tmix_auc", "mix_pauc"),
            multiple_sections=False,
        )
        for year in range(2023, 2027)
    },
}


def average_metrics(values: pd.DataFrame, rules: EvaluationRules) -> pd.Series:
    """Apply the edition's arithmetic or harmonic mean to each row."""
    if rules.aggregation_method == "arithmetic":
        return values.apply(np.mean, axis=1)
    elif rules.aggregation_method == "harmonic":
        return values.apply(hmean, axis=1)
    else:
        raise NotImplementedError(rules.aggregation_method)
