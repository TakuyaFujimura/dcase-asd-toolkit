import numpy as np


def normalize_vector(vector: np.ndarray, eps=1e-12) -> np.ndarray:
    """
    Args:
        vector (np.ndarray): (N, D)

    Returns:
        np.ndarray : normalized vector
    """
    return vector / np.maximum(np.linalg.norm(vector, axis=1, keepdims=True), eps)
