import numpy as np


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """
    Args:
        vector (np.ndarray): (N, D)

    Returns:
        np.ndarray : normalized vector
    """
    return vector / np.linalg.norm(vector, axis=1, keepdims=True)
