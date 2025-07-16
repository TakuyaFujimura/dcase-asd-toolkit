import logging
from typing import Dict, Optional, Tuple

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from asdkit.utils.common.np_util import normalize_vector

from .base import BaseBackend

logger = logging.getLogger(__name__)


class Knn(BaseBackend):
    def __init__(
        self,
        n_neighbors_so: int,
        n_neighbors_ta: int,
        metric: str = "cosine",
        smote_ratio: float = 0,
        smote_neighbors: int = 2,
        sep_section: bool = False,
        embed_key: str = "embed",
    ):
        """
        Args:
            n_neighbors_so (int): Number of neighbors for source domain.
            n_neighbors_ta (int): Number of neighbors for target domain.
            metric (str): Distance metric. Options are "euclid" or "cosine". Defaults to "cosine".
            smote_ratio (float): Ratio of SMOTE sampling applied to the target domain. Defaults to 0 (no SMOTE).
            smote_neighbors (int): Number of neighbors for SMOTE. Defaults to 5.
            sep_section (bool, optional): Whether to separately construct a backend for each section. Defaults to False.
            embed_key (str, optional): Key to access embeddings in the extract_dict. Defaults to "embed".
        """
        if metric not in ["euclid", "cosine"]:
            raise ValueError(f"Unexpected metric: {metric}")
        self.metric = metric
        self.n_neighbors_so = n_neighbors_so
        self.n_neighbors_ta = n_neighbors_ta
        self.sep_section = sep_section
        self.knn_dict: Dict[
            int, Tuple[NearestNeighbors, Optional[NearestNeighbors]]
        ] = {}

        # Instantiate SMOTE
        if smote_ratio == 0:
            self.smote = None
        else:
            self.smote = SMOTE(
                sampling_strategy=smote_ratio,  # type: ignore
                random_state=0,
                k_neighbors=smote_neighbors,
            )
        self.embed_key = embed_key

    def get_section(self, extract_dict: dict) -> np.ndarray:
        section = extract_dict["section"]
        if not self.sep_section:
            section = np.zeros_like(section)
        return section

    def get_knn(
        self, embed: np.ndarray, is_target: np.ndarray
    ) -> Tuple[NearestNeighbors, Optional[NearestNeighbors]]:
        self.check_target(is_target)

        if self.metric == "cosine":
            embed = normalize_vector(embed)

        knn_so = NearestNeighbors(n_neighbors=self.n_neighbors_so, metric="euclidean")
        knn_so.fit(embed[is_target == 0])  # type: ignore

        if np.sum(is_target) > 0 and self.n_neighbors_ta > 0:
            if self.smote is not None:
                embed, is_target = self.smote.fit_resample(embed, is_target)  # type: ignore
                if self.metric == "cosine":
                    embed = normalize_vector(embed)  # type: ignore
            knn_ta = NearestNeighbors(
                n_neighbors=self.n_neighbors_ta, metric="euclidean"
            )
            knn_ta.fit(embed[is_target == 1])  # type: ignore
        else:
            knn_ta = None

        return knn_so, knn_ta

    def fit(self, train_dict: dict) -> None:
        section = self.get_section(train_dict)
        is_target = np.asarray(train_dict["is_target"])
        embed = train_dict[self.embed_key]

        for sec in np.unique(section):
            idx = section == sec
            self.knn_dict[sec] = self.get_knn(
                embed=embed[idx], is_target=is_target[idx]
            )

    def calc_score(
        self,
        embed: np.ndarray,
        knn_list: Tuple[NearestNeighbors, Optional[NearestNeighbors]],
    ) -> np.ndarray:
        knn_so, knn_ta = knn_list
        if self.metric == "cosine":
            embed = normalize_vector(embed)

        scores_so: np.ndarray = knn_so.kneighbors(embed)[0].mean(1)
        if knn_ta is None:
            scores = scores_so
        else:
            scores_ta: np.ndarray = knn_ta.kneighbors(embed)[0].mean(1)
            scores = np.minimum(scores_so, scores_ta)

        if self.metric == "cosine":
            scores /= 2
        return scores

    def anomaly_score(self, test_dict: dict) -> Dict[str, np.ndarray]:
        section = self.get_section(test_dict)
        embed = test_dict[self.embed_key]
        scores = np.zeros(len(section))

        for sec in np.unique(section):
            idx = section == sec
            scores[idx] = self.calc_score(embed[idx], self.knn_dict[sec])

        return {"main": scores}


class KnnRescale(BaseBackend):
    def __init__(
        self,
        n_neighbors: int,
        k_ref_normalize: int = 16,
        metric: str = "cosine",
        sep_section: bool = False,
        embed_key: str = "embed",
    ):
        """
        K. Wilkinghoff et al., "Keeping the Balance: Anomaly Score Calculation for Domain Generalization," Proc. ICASSP, 2025.

        Args:
            n_neighbors (int): Number of neighbors for anomaly score calculation.
            k_ref_normalize (int): Number of neighbors for score normalization. Defaults to 16.
            metric (str): Distance metric. Options are "euclid" or "cosine". Defaults to "cosine".
            sep_section (bool, optional): Whether to separately construct a backend for each section. Defaults to False.
            embed_key (str, optional): Key to access embeddings in the extract_dict. Defaults to "embed".
        """
        if metric not in ["euclid", "cosine"]:
            raise ValueError(f"Unexpected metric: {metric}")
        self.metric = metric
        self.n_neighbors = n_neighbors
        self.sep_section = sep_section
        self.k_ref_normalize = k_ref_normalize
        self.ref_dict: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self.embed_key = embed_key

    def get_section(self, extract_dict: dict) -> np.ndarray:
        section = extract_dict["section"]
        if not self.sep_section:
            section = np.zeros_like(section)
        return section

    def prepare_ref_set(
        self, ref_embed: np.ndarray, is_target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert self.k_ref_normalize > 0
        if self.metric == "cosine":
            ref_embed = normalize_vector(ref_embed)
        ref_norm_const = np.sort(
            pairwise_distances(ref_embed, ref_embed, metric="euclidean"), axis=0
        )[1 : self.k_ref_normalize + 1].mean(axis=0, keepdims=True)
        # (N, N) -> (1, N)
        return ref_embed, ref_norm_const

    def fit(self, train_dict: dict) -> None:
        section = self.get_section(train_dict)
        is_target = np.asarray(train_dict["is_target"])
        embed = train_dict[self.embed_key]

        for sec in np.unique(section):
            idx = section == sec
            self.ref_dict[sec] = self.prepare_ref_set(
                ref_embed=embed[idx], is_target=is_target[idx]
            )

    def calc_score(
        self,
        embed: np.ndarray,
        ref_list: Tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        ref_embed, ref_norm_const = ref_list
        if self.metric == "cosine":
            embed = normalize_vector(embed)
        scores = np.sort(
            pairwise_distances(embed, ref_embed, metric="euclidean") / ref_norm_const,
            axis=1,
        )[:, : self.n_neighbors].mean(axis=1)
        # (N_input, N_ref) -> (N_input, self.n_neighbors_so) -> (N_input,)
        return scores

    def anomaly_score(self, test_dict: dict) -> Dict[str, np.ndarray]:
        section = self.get_section(test_dict)
        embed = test_dict[self.embed_key]
        scores = np.zeros(len(section))

        for sec in np.unique(section):
            idx = section == sec
            scores[idx] = self.calc_score(embed[idx], self.ref_dict[sec])

        return {"main": scores}
