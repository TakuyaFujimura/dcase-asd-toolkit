from typing import Annotated, Any

import torch
from pydantic import PlainSerializer, PlainValidator


def validate(v: Any) -> torch.Tensor:
    if isinstance(v, torch.Tensor):
        ans = v
    elif isinstance(v, (list, tuple)):
        ans = torch.Tensor(v)
    else:
        raise TypeError(
            f"Expected numpy.ndarray, list or tuple of float, got {type(v)}"
        )
    return ans


def serialize(v: torch.Tensor) -> list[float]:
    return v.tolist()


TorchTensor = Annotated[
    torch.Tensor,
    PlainValidator(validate),
    PlainSerializer(serialize),
]
