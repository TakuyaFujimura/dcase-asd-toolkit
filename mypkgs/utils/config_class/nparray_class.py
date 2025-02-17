from typing import Annotated, Any

import numpy as np
from pydantic import PlainSerializer, PlainValidator


def validate(v: Any) -> np.ndarray:
    if isinstance(v, np.ndarray):
        ans = v
    elif isinstance(v, (list, tuple)):
        ans = np.array(v)
    else:
        raise TypeError(
            f"Expected numpy.ndarray, list or tuple of float, got {type(v)}"
        )
    if ans.ndim != 1:
        raise ValueError(f"Expected 1D array, got {ans.ndim}D array")
    return ans


def serialize(v: np.ndarray) -> list[float]:
    return v.tolist()


Np1DArray = Annotated[
    np.ndarray,
    PlainValidator(validate),
    PlainSerializer(serialize),
]
