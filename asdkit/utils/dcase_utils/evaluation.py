# Copyright 2024 Takuya Fujimura

import numpy as np
from sklearn.metrics import roc_auc_score


def get_domain_mask(
    domain: str, is_target: np.ndarray, is_normal: np.ndarray
) -> np.ndarray:
    """
    Args:
        domain (str): One of `s`, `t`, `smix`, `tmix`, or `mix`.
            `s` and `t` select source and target samples, respectively.
            `smix` and `tmix` select normals from the respective domain
            plus anomalies from both domains. `mix` selects all samples.

    Returns:
        np.ndarray: Boolean mask selecting samples for the domain.
    """
    if domain == "s":
        domain_mask = is_target == 0
    elif domain == "t":
        domain_mask = is_target == 1
    elif domain == "smix":
        domain_mask = (is_target == 0) | (is_normal == 0)
    elif domain == "tmix":
        domain_mask = (is_target == 1) | (is_normal == 0)
    elif domain == "mix":
        domain_mask = np.ones_like(is_normal).astype(bool)
    else:
        raise NotImplementedError()
    return domain_mask


def dcase_auc(
    anomaly_score: np.ndarray,
    section_ids: np.ndarray,
    is_normal: np.ndarray,
    is_target: np.ndarray,
    auc_type: str,
) -> float:
    """Compute AUC or standardized partial AUC for one section and domain.

    Args:
        anomaly_score: Per-sample anomaly scores; larger means more anomalous.
        section_ids: Per-sample section IDs.
        is_normal: Per-sample normality flags (1 for normal, 0 for anomalous).
        is_target: Per-sample domain flags (0 for source, 1 for target).
        auc_type: `{section}_{domain}_{metric}`, e.g. `0_smix_pauc`.
            Section is an integer ID, domain is `s`, `t`, `smix`, `tmix`,
            or `mix`, and metric is `auc` or `pauc` (max FPR 0.1).
            Mixed domains include normals from the selected domain and
            anomalies from both domains; `mix` includes both domains entirely.
    """
    is_anomaly = 1 - is_normal

    section_text, domain, metric = auc_type.split("_")
    section_id = int(section_text)
    section_mask = section_ids == section_id
    domain_mask = get_domain_mask(
        domain=domain, is_target=is_target, is_normal=is_normal
    )
    evaluation_mask = section_mask & domain_mask

    if metric == "auc":
        score = roc_auc_score(
            y_true=is_anomaly[evaluation_mask], y_score=anomaly_score[evaluation_mask]
        )
    elif metric == "pauc":
        score = roc_auc_score(
            y_true=is_anomaly[evaluation_mask],
            y_score=anomaly_score[evaluation_mask],
            max_fpr=0.1,
        )
    else:
        raise NotImplementedError()
    return score  # type: ignore
