"""Embedding transformation wrapper"""

from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

from asdkit.models.pretrained_models.views import build_embedding_views
from asdkit.utils.common import instantiate

from ..base import BaseBackend, EmbeddingBackend


class TransformWrapper(EmbeddingBackend):
    """Build configured embedding views, fit children, and aggregate their scores.

    view_name selects CLS, global pooling, frequency grouping, or frequency
    concatenation.
    A single view uses the same scoring path; the aggregation choice then has
    no effect. pooling_name is unused for CLS.
    Saved embeddings with representation="views" are used directly; view_name
    and pooling_name apply only to raw model embeddings.
    Children must implement EmbeddingBackend and expose their embed_key.
    transform_batch_size bounds raw samples per conversion; None converts all.
    Child fitting and distance calculations use all pooled samples.
    """

    def __init__(
        self,
        backend_cfg: Dict[str, Any],
        view_name: Optional[str] = None,
        pooling_name: str = "mean",
        score_aggregation_name: str = "mean",
        embed_key: str = "embed",
        transform_batch_size: Optional[int] = 64,
    ) -> None:
        self.backend_cfg = deepcopy(backend_cfg)
        self.view_name = view_name
        self.pooling_name = pooling_name
        self.score_aggregation_name = score_aggregation_name
        self.embed_key = embed_key
        if transform_batch_size is not None and (
            isinstance(transform_batch_size, bool)
            or not isinstance(transform_batch_size, int)
            or transform_batch_size <= 0
        ):
            raise ValueError("transform_batch_size must be a positive integer or None")
        self.transform_batch_size = transform_batch_size
        self.backends: List[EmbeddingBackend] = []

        if self.score_aggregation_name not in ("mean", "min", "max"):
            raise NotImplementedError(
                f"Unknown score aggregation name: {self.score_aggregation_name}"
            )

    def build_embedding_views(
        self, extract_dict: Dict[str, Any], metadata: Dict[str, Any]
    ) -> np.ndarray:
        """Return (# samples, # views, D') without interpreting sections."""
        embedding_info = metadata.get("embeddings", {}).get(self.embed_key, {})
        representation = embedding_info.get("representation")
        if representation is None:
            raise ValueError(
                f"Missing representation metadata for embedding: {self.embed_key}"
            )
        embeddings = extract_dict[self.embed_key]
        if representation == "views":
            if embeddings.ndim != 3 or embeddings.shape[1] == 0:
                raise ValueError("Saved views must have shape (N, V, D) with V > 0")
            return embeddings
        if self.view_name is None:
            raise ValueError("view_name must be specified for raw model embeddings")

        def build_batch_views(embeddings: np.ndarray) -> np.ndarray:
            # Prepare only this batch for the NumPy-to-PyTorch conversion.
            # PyTorch cannot enforce the read-only flag of a mapped array.
            if not embeddings.flags.writeable:
                embeddings = embeddings.copy()
            return build_embedding_views(
                embeddings=embeddings,
                representation=representation,
                view_name=self.view_name,
                pooling_name=self.pooling_name,
            )

        batch_size = self.transform_batch_size
        if batch_size is None or len(embeddings) <= batch_size:
            return build_batch_views(embeddings=embeddings)
        # Allocate only pooled features. Never concatenate a list of views or
        # retain a CLS view that could keep the raw mapped input alive.
        first = build_batch_views(embeddings=embeddings[:batch_size])
        result = np.empty((len(embeddings), *first.shape[1:]), dtype=first.dtype)
        result[:batch_size] = first
        del first
        for start in range(batch_size, len(embeddings), batch_size):
            result[start : start + batch_size] = build_batch_views(
                embeddings=embeddings[start : start + batch_size]
            )
        return result

    def make_view_data(
        self,
        extract_dict: Dict[str, Any],
        metadata: Dict[str, Any],
        embeddings: np.ndarray,
        embed_key: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        view_dict = dict(extract_dict)
        view_dict[embed_key] = embeddings
        view_metadata = deepcopy(metadata)
        view_metadata.setdefault("embeddings", {})[embed_key] = {
            "representation": "vector"
        }
        return view_dict, view_metadata

    def fit(
        self, train_dict: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        if metadata is None:
            raise ValueError(
                "Metadata must be provided for training the TransformWrapper."
            )
        reference_views = self.build_embedding_views(
            extract_dict=train_dict, metadata=metadata
        )  # (# samples, # views, D')
        assert reference_views.ndim == 3
        assert reference_views.shape[0] == len(train_dict[self.embed_key])
        assert reference_views.shape[1] > 0
        backends: List[EmbeddingBackend] = []
        for view_idx in range(reference_views.shape[1]):
            backend = instantiate(self.backend_cfg)
            if not isinstance(backend, EmbeddingBackend):
                raise TypeError(
                    f"TransformWrapper requires an EmbeddingBackend, got {type(backend).__name__}."
                )
            view_dict, view_metadata = self.make_view_data(
                extract_dict=train_dict,
                metadata=metadata,
                embeddings=reference_views[:, view_idx],
                embed_key=backend.embed_key,
            )
            backend.fit(train_dict=view_dict, metadata=view_metadata)
            backends.append(backend)
        self.backends = backends

    def anomaly_score(
        self,
        extract_dict: Dict[str, Any],
        split: Literal["train", "test"] = "test",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, np.ndarray]:
        if metadata is None:
            raise ValueError(
                "Metadata must be provided for scoring the TransformWrapper."
            )
        # Calculate anomaly scores for each view
        query_views = self.build_embedding_views(
            extract_dict=extract_dict, metadata=metadata
        )  # (# samples, # views, D')
        assert len(self.backends) > 0
        assert query_views.ndim == 3
        assert query_views.shape[0] == len(extract_dict[self.embed_key])
        assert query_views.shape[1] == len(self.backends)
        view_scores_dict: Dict[str, List[np.ndarray]] = {}
        for view_idx, backend in enumerate(self.backends):
            view_dict, view_metadata = self.make_view_data(
                extract_dict=extract_dict,
                metadata=metadata,
                embeddings=query_views[:, view_idx],
                embed_key=backend.embed_key,
            )
            scores_dict = backend.anomaly_score(
                extract_dict=view_dict, split=split, metadata=view_metadata
            )
            if view_idx == 0:
                view_scores_dict = {key: [] for key in scores_dict}
            assert scores_dict.keys() == view_scores_dict.keys()
            for key in scores_dict:
                assert scores_dict[key].shape == (query_views.shape[0],)
                view_scores_dict[key].append(scores_dict[key])

        # Aggregate scores across views
        if self.score_aggregation_name == "mean":
            aggregate = np.mean
        elif self.score_aggregation_name == "min":
            aggregate = np.min
        elif self.score_aggregation_name == "max":
            aggregate = np.max
        else:
            raise NotImplementedError(
                f"Unknown score aggregation name: {self.score_aggregation_name}"
            )
        return {
            key: aggregate(np.stack(scores_list, axis=1), axis=1)
            for key, scores_list in view_scores_dict.items()
        }


class BEAM(TransformWrapper):
    """
    P. Saengthong et al., "Sub-Band Spectral Matching with Localized Score Aggregation for Robust Anomalous Sound Detection,"
    """

    def __init__(
        self,
        backend_cfg: Dict[str, Any],
        embed_key: str = "embed",
        transform_batch_size: Optional[int] = 64,
    ) -> None:
        super().__init__(
            backend_cfg=backend_cfg,
            view_name="freq_grouped_time_pooled",
            pooling_name="mean",
            score_aggregation_name="mean",
            embed_key=embed_key,
            transform_batch_size=transform_batch_size,
        )


class AdaBEAM(BaseBackend):
    """
    P. Saengthong et al., "Sub-Band Spectral Matching with Localized Score Aggregation for Robust Anomalous Sound Detection,"

    Average mean-pooled and max-pooled TransformWrapper scores.
    """

    def __init__(
        self,
        backend_cfg: Dict[str, Any],
        embed_key: str = "embed",
        transform_batch_size: Optional[int] = 64,
    ) -> None:
        self.backends: List[TransformWrapper] = [
            TransformWrapper(
                backend_cfg=backend_cfg,
                view_name="freq_grouped_time_pooled",
                pooling_name=pooling_name,
                score_aggregation_name="mean",
                embed_key=embed_key,
                transform_batch_size=transform_batch_size,
            )
            for pooling_name in ("mean", "max")
        ]

    def fit(
        self, train_dict: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        for backend in self.backends:
            backend.fit(train_dict=train_dict, metadata=metadata)

    def anomaly_score(
        self,
        extract_dict: Dict[str, Any],
        split: Literal["train", "test"] = "test",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, np.ndarray]:
        scores = [
            backend.anomaly_score(
                extract_dict=extract_dict, split=split, metadata=metadata
            )
            for backend in self.backends
        ]
        assert scores[0].keys() == scores[1].keys()
        return {key: (scores[0][key] + scores[1][key]) / 2 for key in scores[0]}
