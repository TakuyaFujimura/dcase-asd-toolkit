"""KNN backends"""

import logging
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from asdkit.utils.common.np_util import normalize_vector

from .base import EmbeddingBackend

logger = logging.getLogger(__name__)

SKLEARN_METRICS: Dict[str, str] = {"euclid": "euclidean", "cosine": "cosine"}


def get_sklearn_metric(metric: str) -> str:
    if metric not in SKLEARN_METRICS:
        raise ValueError(f"Unexpected metric: {metric}")
    return SKLEARN_METRICS[metric]


def calculate_pairwise_distances(
    query_embeddings: np.ndarray, reference_embeddings: np.ndarray, metric: str
) -> np.ndarray:
    """
    Args:
        query_embeddings: (# query samples, embedding dim)
        reference_embeddings: (# reference samples, embedding dim)
        metric: distance metric
    Return:
        dists: (# query samples, # reference samples)
    """
    assert query_embeddings.ndim == 2
    assert reference_embeddings.ndim == 2
    assert query_embeddings.shape[1] == reference_embeddings.shape[1]
    dists = pairwise_distances(query_embeddings, reference_embeddings, metric=metric)
    return dists


def build_nearest_neighbors(
    reference_embeddings: np.ndarray, metric: str
) -> NearestNeighbors:
    assert reference_embeddings.ndim == 2
    knn = NearestNeighbors(metric=metric)
    knn.fit(reference_embeddings)
    return knn


def remove_self_neighbor_distances(dists: np.ndarray) -> np.ndarray:
    assert dists.ndim == 2
    assert dists.shape[1] > 1
    return dists[:, 1:]


def mask_self_distances(dists: np.ndarray) -> np.ndarray:
    assert dists.ndim == 2
    assert dists.shape[0] == dists.shape[1]
    np.fill_diagonal(dists, np.inf)
    return dists


def mean_neighbor_distance(
    embed: np.ndarray,
    knn: NearestNeighbors,
    n_neighbors: int,
    exclude_self: bool = False,
) -> np.ndarray:
    """Return mean neighbor distances, preserving empty query batches.

    With exclude_self=True, queries must be fitted samples; the closest
    neighbor is removed before averaging.
    """
    assert embed.ndim == 2
    if len(embed) == 0:
        return np.empty(0, dtype=float)

    n_neighbors += int(exclude_self)
    distances = knn.kneighbors(embed, n_neighbors=n_neighbors)[0]
    if exclude_self:
        distances = remove_self_neighbor_distances(distances)
    return distances.mean(axis=1)


class KnnBackend(EmbeddingBackend):
    def __init__(
        self, n_neighbors: int = 1, metric: str = "cosine", embed_key: str = "embed"
    ) -> None:
        """
        Args:
            n_neighbors (int): Number of neighbors for anomaly score calculation.
            metric (str): Distance metric. Options are "euclid" or "cosine". Defaults to "cosine".
            embed_key (str, optional): Key to access embeddings in the extract_dict. Defaults to "embed".
        """
        self.metric = get_sklearn_metric(metric=metric)
        assert n_neighbors > 0
        self.n_neighbors = n_neighbors
        self.reference: Optional[NearestNeighbors] = None
        self.embed_key = embed_key

    def _prepare_reference(self, train_dict: Dict[str, Any]) -> np.ndarray:
        """Use all embeddings as references without consulting domain labels."""
        embeddings = np.asarray(train_dict[self.embed_key], dtype=np.float64)
        assert embeddings.ndim == 2
        return embeddings

    def fit(
        self, train_dict: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Build a neighbor memory from the prepared reference embeddings."""
        embeddings = self._prepare_reference(train_dict=train_dict)
        self.reference = build_nearest_neighbors(
            reference_embeddings=embeddings, metric=self.metric
        )

    def anomaly_score(
        self,
        extract_dict: Dict[str, Any],
        split: Literal["train", "test"] = "test",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, np.ndarray]:
        """Return mean neighbor distances for embeddings shaped (B, D).

        For split="train", queries must be fitted samples so the closest
        neighbor can be excluded as self. Returns main scores shaped (B,).
        """
        assert self.reference is not None
        embeddings = np.asarray(extract_dict[self.embed_key], dtype=np.float64)
        assert embeddings.ndim == 2
        scores = mean_neighbor_distance(
            embed=embeddings,
            knn=self.reference,
            n_neighbors=self.n_neighbors,
            exclude_self=split == "train",
        )
        return {"main": scores}


class KnnSMOTEBackend(KnnBackend):
    def __init__(
        self,
        n_neighbors: int = 1,
        metric: str = "cosine",
        smote_ratio: float = 0.2,
        smote_neighbors: int = 2,
        embed_key: str = "embed",
    ) -> None:
        """Augment target references with seeded SMOTE.

        smote_ratio is the desired target/source sample ratio and
        smote_neighbors controls neighbors used to generate target samples.
        When targets exist, the target domain must be the minority class and
        smote_ratio > 0. Without targets, use standard kNN without SMOTE.
        """
        super().__init__(n_neighbors=n_neighbors, metric=metric, embed_key=embed_key)

        assert smote_ratio > 0
        self.smote = SMOTE(
            sampling_strategy=smote_ratio,  # type: ignore
            random_state=0,
            k_neighbors=smote_neighbors,
        )

    def _prepare_reference(self, train_dict: Dict[str, Any]) -> np.ndarray:
        """Combine source, original target, and synthetic target references."""
        embeddings = super()._prepare_reference(train_dict)
        is_target = np.asarray(train_dict["is_target"])
        assert len(embeddings) == len(is_target)
        if not np.any(is_target == 1):
            logger.warning(
                "No target samples found. Skipping SMOTE and using standard kNN."
            )
            return embeddings
        assert np.sum(is_target == 1) < np.sum(is_target == 0)

        if self.metric == "cosine":
            # To apply SMOTE in cosine space
            embeddings = normalize_vector(embeddings)
        embeddings, _ = self.smote.fit_resample(embeddings, is_target)  # type: ignore
        return embeddings


class KnnKmeansBackend(KnnBackend):
    def __init__(
        self,
        n_neighbors: int = 1,
        n_clusters: int = 16,
        metric: str = "cosine",
        embed_key: str = "embed",
    ) -> None:
        """Replace source references with n_clusters centers and retain targets."""
        super().__init__(n_neighbors=n_neighbors, metric=metric, embed_key=embed_key)
        assert n_clusters > 0
        self.n_clusters = n_clusters

    def _prepare_reference(self, train_dict: Dict[str, Any]) -> np.ndarray:
        """Cluster source embeddings with a fixed seed, then append targets."""
        embeddings = super()._prepare_reference(train_dict=train_dict)
        is_target = np.asarray(train_dict["is_target"])
        assert len(embeddings) == len(is_target)
        if np.sum(is_target) == 0:
            logger.warning("number of target data is 0.")
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0)
        if self.metric == "cosine":
            # Normalize before Euclidean KMeans to approximate cosine clustering.
            # This differs from spherical KMeans, which also normalizes centers.
            embeddings = normalize_vector(embeddings)
        kmeans.fit(embeddings[is_target == 0])
        return np.vstack([kmeans.cluster_centers_, embeddings[is_target == 1]])

    def anomaly_score(
        self,
        extract_dict: Dict[str, Any],
        split: Literal["train", "test"] = "test",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, np.ndarray]:
        """Exclude self only for retained target samples when scoring training data.

        Source samples were replaced by centers and have no self reference.
        Training queries must be fitted samples with matching is_target flags.
        Test queries need only embeddings.
        """
        if split == "test":
            return super().anomaly_score(
                extract_dict=extract_dict, split=split, metadata=metadata
            )

        assert split == "train"
        assert self.reference is not None
        embeddings = np.asarray(extract_dict[self.embed_key], dtype=np.float64)
        assert embeddings.ndim == 2
        is_target = np.asarray(extract_dict["is_target"])
        assert len(embeddings) == len(is_target)
        assert np.all((is_target == 0) | (is_target == 1))

        scores = np.empty(len(embeddings), dtype=float)
        for is_target_id, exclude_self in ((0, False), (1, True)):
            idx = is_target == is_target_id
            scores[idx] = mean_neighbor_distance(
                embed=embeddings[idx],
                knn=self.reference,
                n_neighbors=self.n_neighbors,
                exclude_self=exclude_self,
            )
        return {"main": scores}


def compute_local_scale(
    embeddings: np.ndarray, k_ref_normalize: int, metric: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute local distance scale within reference samples"""
    assert 0 < k_ref_normalize < len(embeddings)

    ref_knn = build_nearest_neighbors(reference_embeddings=embeddings, metric=metric)
    dists, indices = ref_knn.kneighbors(embeddings, n_neighbors=k_ref_normalize + 1)
    # bias (N_ref, k + 1) -> (N_ref, k) -> (N_ref,)
    ref_local_scale = remove_self_neighbor_distances(dists).mean(axis=1)
    # ref_local_scale: (N_ref,), broadcast across query rows.
    if np.any(ref_local_scale == 0):
        raise ValueError("Some reference samples have zero mean neighbor distance")
    return dists, indices, ref_local_scale


class KnnRescaleBackend(EmbeddingBackend):
    def __init__(
        self,
        n_neighbors: int,
        k_ref_normalize: int = 4,
        metric: str = "cosine",
        embed_key: str = "embed",
    ) -> None:
        """
        K. Wilkinghoff et al., "Keeping the Balance: Anomaly Score Calculation for Domain Generalization," Proc. ICASSP, 2025.

        Args:
            n_neighbors (int): Number of neighbors for anomaly score calculation.
            k_ref_normalize (int): Number of neighbors for score normalization. Defaults to 4.
            metric (str): Distance metric. Options are "euclid" or "cosine". Defaults to "cosine".
            embed_key (str, optional): Key to access embeddings in the extract_dict. Defaults to "embed".
        """
        self.metric = get_sklearn_metric(metric=metric)
        self.n_neighbors = n_neighbors
        self.k_ref_normalize = k_ref_normalize
        self.reference: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.embed_key = embed_key

    def fit(
        self, train_dict: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store reference embeddings and their local distance scales."""
        embeddings = np.asarray(train_dict[self.embed_key], dtype=np.float64)
        assert embeddings.ndim == 2
        _, _, ref_norm_const = compute_local_scale(
            embeddings=embeddings,
            k_ref_normalize=self.k_ref_normalize,
            metric=self.metric,
        )
        self.reference = embeddings, ref_norm_const[None, :]

    def anomaly_score(
        self,
        extract_dict: Dict[str, Any],
        split: Literal["train", "test"] = "test",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, np.ndarray]:
        """Return corrected neighbor-distance scores shaped (B,) under main.

        extract_dict contains embeddings shaped (B, D).
        """
        assert self.reference is not None
        embeddings = np.asarray(extract_dict[self.embed_key], dtype=np.float64)
        assert embeddings.ndim == 2
        ref_embed, ref_norm_const = self.reference
        distances = calculate_pairwise_distances(
            query_embeddings=embeddings,
            reference_embeddings=ref_embed,
            metric=self.metric,
        )
        if split == "train":
            distances = mask_self_distances(distances)
        # (N_query, N_ref)
        scores = np.sort(distances / ref_norm_const, axis=1)[
            :, : self.n_neighbors
        ].mean(axis=1)
        # (N_input, N_ref) -> (N_input, self.n_neighbors) -> (N_input,)
        return {"main": np.asarray(scores, dtype=float)}


class KnnRescaleVarMinBackend(EmbeddingBackend):
    def __init__(
        self,
        n_neighbors: int,
        k_ref_normalize: int = 4,
        metric: str = "cosine",
        embed_key: str = "embed",
    ) -> None:
        """
        M. Matsumoto et al., "Adjusting Bias in Anomaly Scores via Variance Minimization for Domain-Generalized Discriminative Anomalous Sound Detection," Proc. DCASE Workshop, 2025.

        Args:
            n_neighbors (int): Number of neighbors for anomaly score calculation.
            k_ref_normalize (int): Number of neighbors for score normalization. Defaults to 4.
            metric (str): Distance metric. Options are "euclid" or "cosine". Defaults to "cosine".
            embed_key (str, optional): Key to access embeddings in the extract_dict. Defaults to "embed".
        """
        self.metric = get_sklearn_metric(metric=metric)
        self.n_neighbors = n_neighbors
        self.k_ref_normalize = k_ref_normalize
        self.reference: Optional[Tuple[np.ndarray, np.ndarray, float]] = None
        self.embed_key = embed_key

    def fit(
        self, train_dict: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Estimate local scales and the variance-minimizing correction coefficient."""
        embeddings = np.asarray(train_dict[self.embed_key], dtype=np.float64)
        assert embeddings.ndim == 2
        assert 0 < self.k_ref_normalize < len(embeddings) - 1
        dists, indices, ref_norm_const = compute_local_scale(
            embeddings=embeddings,
            k_ref_normalize=self.k_ref_normalize,
            metric=self.metric,
        )

        # Nearest non-self neighbor of each reference sample (N_ref)
        d_opt_indices = indices[:, 1]
        d_opt = dists[:, 1]

        # Bias of the selected neighbors (N_ref)
        b_opt = ref_norm_const[d_opt_indices]

        # Optimal alpha for variance minimization
        b_opt_var = np.var(b_opt)
        if b_opt_var == 0:
            logger.warning("Variance of bias is zero. Setting alpha to 0.0.")
            alpha = 0.0
        else:
            # Minimize Var(d_opt - alpha * b_opt)
            alpha = np.cov(d_opt, b_opt, bias=True)[0, 1] / b_opt_var

        self.reference = embeddings, ref_norm_const, float(alpha)

    def anomaly_score(
        self,
        extract_dict: Dict[str, Any],
        split: Literal["train", "test"] = "test",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, np.ndarray]:
        """Return corrected neighbor-distance scores shaped (B,) under main.

        extract_dict contains embeddings shaped (B, D). For split="train",
        queries must be the fitted samples in their original order so the
        reference distance diagonal identifies self-neighbors.
        """
        assert self.reference is not None
        embeddings = np.asarray(extract_dict[self.embed_key], dtype=np.float64)
        assert embeddings.ndim == 2
        ref_embed, ref_norm_const, alpha = self.reference
        distances = calculate_pairwise_distances(
            query_embeddings=embeddings,
            reference_embeddings=ref_embed,
            metric=self.metric,
        )
        if split == "train":
            distances = mask_self_distances(distances)
        scores = np.sort(distances - alpha * ref_norm_const, axis=1)[
            :, : self.n_neighbors
        ].mean(axis=1)
        # (N_input, N_ref) -> (N_input, self.n_neighbors) -> (N_input,)
        return {"main": np.asarray(scores, dtype=float)}
