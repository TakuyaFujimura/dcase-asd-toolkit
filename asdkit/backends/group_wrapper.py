"""Fit independent backends per sample label without changing sample order."""

from copy import deepcopy
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple

import numpy as np

from asdkit.utils.common import instantiate

from .base import BaseBackend

GroupKey = Tuple[Any, ...]


def iter_label_groups(
    extract_dict: Dict[str, Any], groupby: Optional[str] = None
) -> Iterator[Tuple[GroupKey, np.ndarray]]:
    """Yield label tuples and row indices, preserving order inside each group.

    All extract_dict values are sample arrays with the same leading dimension.
    Grouping labels are one-dimensional scalar arrays; metadata is separate.
    None yields one group containing all sample rows.
    """
    n_samples = len(next(iter(extract_dict.values())))
    assert all(len(value) == n_samples for value in extract_dict.values())
    if groupby is None:
        yield (), np.arange(n_samples)
        return

    labels = np.asarray(extract_dict[groupby])
    assert labels.shape == (n_samples,)
    indices_by_label: Dict[GroupKey, List[int]] = {}
    for index in range(n_samples):
        key = (labels[index],)
        indices_by_label.setdefault(key, []).append(index)
    for key, indices in indices_by_label.items():
        yield key, np.asarray(indices, dtype=np.intp)


class GroupWrapper(BaseBackend):
    """Wrap any backend with independent fitting for each label value.

    Groups default to section; None uses all samples.
    backend_cfg is instantiated once per group during fit. Metadata is forwarded
    separately. The train split retains the child's self-exclusion contract:
    provide the fitted samples in the same order within each group.
    """

    def __init__(
        self, backend_cfg: Dict[str, Any], groupby: Optional[str] = "section"
    ) -> None:
        self.backend_cfg = deepcopy(backend_cfg)
        self.groupby = groupby
        self.backends: Dict[GroupKey, BaseBackend] = {}

    def fit(
        self, train_dict: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        backends: Dict[GroupKey, BaseBackend] = {}
        for key, indices in iter_label_groups(
            extract_dict=train_dict, groupby=self.groupby
        ):
            group_dict = {
                name: np.asarray(values)[indices] for name, values in train_dict.items()
            }
            backend = instantiate(self.backend_cfg)
            backend.fit(train_dict=group_dict, metadata=metadata)
            backends[key] = backend
        self.backends = backends

    def anomaly_score(
        self,
        extract_dict: Dict[str, Any],
        split: Literal["train", "test"] = "test",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, np.ndarray]:
        assert self.backends
        n_samples = len(next(iter(extract_dict.values())))
        result: Dict[str, np.ndarray] = {}
        for key, indices in iter_label_groups(
            extract_dict=extract_dict, groupby=self.groupby
        ):
            if key not in self.backends:
                raise ValueError(f"Unseen label group: {key}")
            group_dict = {
                name: np.asarray(values)[indices]
                for name, values in extract_dict.items()
            }
            scores = self.backends[key].anomaly_score(
                extract_dict=group_dict, split=split, metadata=metadata
            )
            if not result:
                result = {
                    name: np.empty(n_samples, dtype=values.dtype)
                    for name, values in scores.items()
                }
            assert scores.keys() == result.keys()
            for name, values in scores.items():
                assert values.shape == (len(indices),)
                result[name][indices] = values
        return result
