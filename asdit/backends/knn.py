import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances

from asdit.utils.common import get_embed_from_df

from .base import BaseBackend
from .utils import normalize_vector

logger = logging.getLogger(__name__)


class Knn(BaseBackend):
    def __init__(
        self,
        metric: str,
        n_neighbors_so: int,
        n_neighbors_ta: int,
        smote_ratio: float = 0,
        smote_neighbors: int = 5,
        sep_section: bool = False,
        k_ref_normalize: int = 0,
    ):
        if metric not in ["euclid", "cosine"]:
            raise ValueError(f"Unexpected metric: {metric}")
        self.metric = metric
        self.n_neighbors_so = n_neighbors_so
        self.n_neighbors_ta = n_neighbors_ta
        self.smote_ratio = smote_ratio
        self.smote_neighbors = smote_neighbors
        self.sep_section = sep_section
        if k_ref_normalize == 0:
            self.k_ref_normalize = None
        else:
            self.k_ref_normalize = k_ref_normalize
            self.ref_dict: Dict[
                int, Tuple[np.ndarray, np.ndarray]
            ] = {}

        self.knn_dict: Dict[
            int, Tuple[NearestNeighbors, Optional[NearestNeighbors]]
        ] = {}
        if self.smote_ratio == 0:
            self.smote = None
        else:
            self.smote = SMOTE(
                sampling_strategy=self.smote_ratio,  # type: ignore
                random_state=0,
                k_neighbors=self.smote_neighbors,
            )

    def get_knn(
        self, train_df: pd.DataFrame
    ) -> Tuple[NearestNeighbors, Optional[NearestNeighbors]]:
        is_target = np.asarray(train_df["is_target"].values)
        self.check_target(is_target)
        embed = get_embed_from_df(train_df)

        if self.metric == "cosine":
            embed = normalize_vector(embed)

        knn_so = NearestNeighbors(n_neighbors=self.n_neighbors_so, metric="euclidean")
        knn_so.fit(embed[is_target == 0])  # type: ignore

        if np.sum(is_target) > 0:
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

    def prepare_ref_set(self, train_df: pd.DataFrame) -> None:
        is_target = np.asarray(train_df["is_target"].values)
        self.check_target(is_target)
        ref_embed = get_embed_from_df(train_df)
        if self.metric == "cosine":
            ref_embed = normalize_vector(ref_embed)
        ref_norm_const = np.sort(pairwise_distances(ref_embed, ref_embed, metric=self.metric), axis=0)[1:self.k_ref_normalize+1].mean(axis=0, keepdims=True)
        return ref_embed, ref_norm_const

    def fit(self, train_df: pd.DataFrame) -> None:
        section = train_df["section"].values
        if self.sep_section:
            for sec in np.unique(section):  # type: ignore
                if self.k_ref_normalize is None:
                    self.knn_dict[sec] = self.get_knn(train_df[section == sec])
                else:
                    self.ref_dict[sec] = self.prepare_ref_set(train_df[section == sec])
        else:
            if self.k_ref_normalize is None:
                self.knn_dict[0] = self.get_knn(train_df)
            else:
                self.ref_dict[0] = self.prepare_ref_set(train_df)

    def calc_score(
        self,
        test_df: pd.DataFrame,
        knn_list: Tuple[NearestNeighbors, Optional[NearestNeighbors]],
    ) -> np.ndarray:
        knn_so, knn_ta = knn_list
        embed = get_embed_from_df(test_df)
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

    def calc_normalized_score(
        self,
        test_df: pd.DataFrame,
        ref_list: Tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        ref_embed, ref_norm_const = ref_list
        embed = get_embed_from_df(test_df)
        if self.metric == "cosine":
            embed = normalize_vector(embed)
        scores = np.sort(pairwise_distances(embed, ref_embed, metric=self.metric)/ref_norm_const, axis=1)[0:self.n_neighbors_so].mean(axis=1)
        return scores

    def anomaly_score(self, test_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        section = test_df["section"].values
        scores = np.zeros(len(test_df))
        if self.sep_section:
            for sec in np.unique(section):  # type: ignore
                if self.k_ref_normalize is None:
                    scores[section == sec] = self.calc_score(
                        test_df[section == sec], self.knn_dict[sec]
                    )
                else:
                    scores[section == sec] = self.calc_normalized_score(
                        test_df[section == sec], self.ref_dict[sec]
                    )
        else:
            if self.k_ref_normalize is None:
                scores = self.calc_score(test_df, self.knn_dict[0])
            else:
                scores = self.calc_normalized_score(test_df, self.ref_dict[0])

        return {"plain": scores}
