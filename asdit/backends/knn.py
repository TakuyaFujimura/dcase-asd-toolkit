import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import NearestNeighbors

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
    ):
        if metric not in ["euclid", "cosine"]:
            raise ValueError(f"Unexpected metric: {metric}")
        self.metric = metric
        self.n_neighbors_so = n_neighbors_so
        self.n_neighbors_ta = n_neighbors_ta
        self.smote_ratio = smote_ratio
        self.smote_neighbors = smote_neighbors
        self.sep_section = sep_section

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

    def fit(self, train_df: pd.DataFrame) -> None:
        section = train_df["section"].values
        if self.sep_section:
            for sec in np.unique(section):  # type: ignore
                self.knn_dict[sec] = self.get_knn(train_df[section == sec])
        else:
            self.knn_dict[0] = self.get_knn(train_df)

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

    def anomaly_score(self, test_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        section = test_df["section"].values
        scores = np.zeros(len(test_df))
        if self.sep_section:
            for sec in np.unique(section):  # type: ignore
                scores[section == sec] = self.calc_score(
                    test_df[section == sec], self.knn_dict[sec]
                )
        else:
            scores = self.calc_score(test_df, self.knn_dict[0])

        return {"plain": scores}
