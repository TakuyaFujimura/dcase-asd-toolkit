"""Convert saved configuration to values supported by weights-only loading."""

from collections.abc import Mapping
from enum import Enum
from pathlib import PurePath
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from pydantic import BaseModel


def to_checkpoint_config(value: Any) -> Any:
    """Resolve configuration containers and remove Python-specific value types."""
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, BaseModel):
        value = value.model_dump()
    if isinstance(value, Enum):
        return to_checkpoint_config(value.value)
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, np.ndarray):
        return to_checkpoint_config(value.tolist())
    if isinstance(value, np.generic):
        return to_checkpoint_config(value.item())
    if isinstance(value, Mapping):
        return {
            to_checkpoint_config(key): to_checkpoint_config(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [to_checkpoint_config(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_checkpoint_config(item) for item in value)
    if value is None or isinstance(value, (str, bool, int, float, torch.Tensor)):
        return value
    raise TypeError(
        f"Unsupported checkpoint configuration type: {type(value).__name__}"
    )
