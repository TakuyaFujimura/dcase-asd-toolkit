"""Convert raw model embeddings into one or more vector views per sample."""

from functools import wraps
from importlib import import_module
from typing import TypeVar, get_args

import numpy as np
import torch

from asdkit.models.pooling import Axis, rdp_pool
from asdkit.utils.common.representation import EmbeddingRepresentation

Array = TypeVar("Array", np.ndarray, torch.Tensor)


def _preserve_array_type(function):
    """Compute on tensors; return the input array type without moving tensors."""

    @wraps(function)
    def wrapped(embeddings, *args, **kwargs):
        is_numpy = isinstance(embeddings, np.ndarray)
        if is_numpy:
            embeddings = torch.as_tensor(embeddings)
        result = function(embeddings, *args, **kwargs)
        return result.numpy() if is_numpy else result

    return wrapped


@_preserve_array_type
def pool_embeddings(embeddings: Array, axis: Axis, pooling_name: str) -> Array:
    if pooling_name == "mean":
        return embeddings.mean(dim=axis)
    if pooling_name == "max":
        return torch.amax(embeddings, dim=axis)
    if pooling_name.startswith("rdp_"):
        parts = pooling_name.split("_")
        if len(parts) != 2:
            raise ValueError(f"Invalid RDP pooling name: {pooling_name}")
        return rdp_pool(embeddings, gamma=float(parts[1]), pooling_axis=axis)
    raise NotImplementedError(f"Unknown pooling name: {pooling_name}")


@_preserve_array_type
def convert_embedding_layout(
    embeddings: Array, representation: str, output_mode: str
) -> Array:
    """Convert raw SSL tokens to time-frequency features or a CLS vector."""
    # Check representation and embedding shape
    if representation not in get_args(EmbeddingRepresentation):
        raise ValueError(f"Unknown representation: {representation}")
    if representation in ("vector", "views"):
        raise ValueError(f"Representation {representation} is not raw SSL features.")
    if embeddings.ndim != 3:
        raise ValueError(
            f"Representation {representation} is expected to have 3 dimensions, but got {embeddings.ndim}."
        )

    # Apply the model-specific feature transform
    module = import_module(f"asdkit.models.pretrained_models.{representation}")
    return module.transform_features(features=embeddings, output_mode=output_mode)


@_preserve_array_type
def build_embedding_views(
    embeddings: Array,
    representation: str,
    view_name: str,
    pooling_name: str = "mean",
) -> Array:
    """Return (N, V, D) vectors; CLS does not use pooling_name."""
    if view_name == "cls":
        embeddings = convert_embedding_layout(
            embeddings=embeddings, representation=representation, output_mode="cls"
        )
        # Return CLS token as a single group
        return embeddings[:, None, :]  # (b, 1, d)

    if view_name not in (
        "freq_grouped_time_pooled",
        "freq_concat_time_pooled",
        "global_pooled",
    ):
        raise ValueError(f"Unknown embedding view: {view_name}")

    embeddings = convert_embedding_layout(
        embeddings=embeddings,
        representation=representation,
        output_mode="time_frequency",
    )
    if view_name == "freq_grouped_time_pooled":
        # Pool time: (b, t, f, d) -> (b, f, d)
        return pool_embeddings(embeddings=embeddings, axis=1, pooling_name=pooling_name)
    elif view_name == "freq_concat_time_pooled":
        b, t, f, d = embeddings.shape
        pooled_embeddings = pool_embeddings(
            embeddings=embeddings.reshape(b, t, f * d),
            axis=1,
            pooling_name=pooling_name,
        )
        return pooled_embeddings[:, None, :]  # (b, 1, f * d)
    else:
        assert view_name == "global_pooled"
        pooled_embeddings = pool_embeddings(
            embeddings=embeddings, axis=(1, 2), pooling_name=pooling_name
        )
        return pooled_embeddings[:, None, :]  # (b, 1, d)
