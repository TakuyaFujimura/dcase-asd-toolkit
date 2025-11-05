import numpy as np


def get_domain_idx(
    auc_type: str, is_target: np.ndarray, is_normal: np.ndarray
) -> np.ndarray:
    """
    Args:
        auc_type (str): in format `{section}_{domain}_{auc/pauc}`
        (e.g., 0_s_auc, 0_tmix_pauc)
    """
    domain = auc_type.split("_")[1]
    if domain == "s":
        domain_idx = is_target == 0
    elif domain == "t":
        domain_idx = is_target == 1
    elif domain == "smix":
        domain_idx = (is_target == 0) | (is_normal == 0)
    elif domain == "tmix":
        domain_idx = (is_target == 1) | (is_normal == 0)
    elif domain == "mix":
        domain_idx = np.ones_like(is_normal).astype(bool)
    else:
        raise NotImplementedError()
    return domain_idx


def get_dcase_idx(
    auc_type: str, section: np.ndarray, is_target: np.ndarray, is_normal: np.ndarray
) -> np.ndarray:
    """
    Args:
        auc_type (str): in format `{section}_{domain}_{auc/pauc}`
        Actually, `auc/pauc` is not used in this function.
        (e.g., 0_s_auc, 0_tmix_pauc)

        section (np.ndarray): section
        is_target (np.ndarray): is_target
        is_normal (np.ndarray): is_normal

    Returns:
        np.ndarray: boolean index for the given auc_type
    """
    domain_idx = get_domain_idx(
        auc_type=auc_type, is_target=is_target, is_normal=is_normal
    )
    section_idx = section == int(auc_type.split("_")[0])
    return domain_idx & section_idx
