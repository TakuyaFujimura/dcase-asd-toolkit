"""Pooling operations on tensors"""

from typing import Tuple, Union

import torch

Axis = Union[int, Tuple[int, ...]]


def rdp_pool(
    x: torch.Tensor,
    gamma: float,
    pooling_axis: Axis,
    dim_axis: int = -1,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Relative Deviation Pooling (RDP) for temporal embedding sequences

    K. Wilkinghoff et al., "Temporal Pooling Strategies for Training-Free Anomalous Sound Detection with Self-Supervised Audio Embeddings"

    Args:
        x: Input embeddings of shape (..., T, D)
    """
    if gamma < 0:
        raise ValueError(f"gamma must be >= 0, got {gamma}")
    if not isinstance(dim_axis, int):
        raise ValueError(f"dim_axis must be an int, got {type(dim_axis)}")
    x_mean = x.mean(dim=pooling_axis, keepdim=True)
    dist = torch.linalg.vector_norm(x - x_mean, dim=dim_axis, keepdim=True)
    dist_max = torch.amax(dist, dim=pooling_axis, keepdim=True)
    rel_dist = dist / dist_max.clamp_min(eps)
    weights = (1.0 + rel_dist) ** gamma
    weights = weights / weights.sum(dim=pooling_axis, keepdim=True)
    return (weights * x).sum(dim=pooling_axis)
