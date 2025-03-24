import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from asdit.utils.common import get_embed_from_df

from .base import BaseBackend
from .utils import normalize_vector

logger = logging.getLogger(__name__)


def min_squared_dist(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Args:
        x (np.ndarray): (N, D)
        y (np.ndarray): (M, D)

    Returns:
        np.ndarray : minimum squared distance from x to y
    """
    squared_dist = np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    # (N, 1, D) - (1, M, D) -> (N, M, D) -> (N, M)
    return np.min(squared_dist, axis=-1)  # (N,)


def min_euclid(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    min_euclid_distance = np.sqrt(min_squared_dist(x, y))
    return min_euclid_distance


def min_cosine(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = normalize_vector(x)
    y = normalize_vector(y)
    min_cosine_distance = min_squared_dist(x, y) / 2
    return min_cosine_distance


class Kmeans(BaseBackend):
    def __init__(self, n_clusters: int, metric: str, sep_section: bool = False):
        if metric not in ["euclid", "cosine"]:
            logger.error(f"Unexpected metric: {metric}")
            raise ValueError(f"Unexpected metric: {metric}")
        self.n_clusters = n_clusters
        self.metric = metric
        self.sep_section = sep_section
        self.centers_dict: Dict[int, np.ndarray] = {}

    def get_centers(self, train_df: pd.DataFrame) -> np.ndarray:
        is_target = np.asarray(train_df["is_target"].values)
        self.check_target(is_target)
        embed = get_embed_from_df(train_df)

        if self.metric == "cosine":
            embed = normalize_vector(embed)

        centers_ta: np.ndarray = embed[is_target == 1]  # (M, D)
        kmeans_so = KMeans(n_clusters=self.n_clusters, random_state=0)
        kmeans_so.fit(embed[is_target == 0])
        centers_so: np.ndarray = kmeans_so.cluster_centers_

        if self.metric == "cosine":
            centers_so = normalize_vector(centers_so)

        centers = np.vstack([centers_so, centers_ta])  # (M, D)
        return centers

    def fit(self, train_df: pd.DataFrame):
        section = train_df["section"].values
        if self.sep_section:
            for sec in np.unique(section):  # type: ignore
                self.centers_dict[sec] = self.get_centers(train_df[section == sec])
        else:
            self.centers_dict[0] = self.get_centers(train_df)

    def calc_score(self, test_df: pd.DataFrame, centers: np.ndarray) -> np.ndarray:
        embed = get_embed_from_df(test_df)  # (N, D)
        if self.metric == "cosine":
            scores = min_cosine(embed, centers)  # (N,)
        else:
            scores = min_euclid(embed, centers)  # (N,)
        return scores

    def anomaly_score(self, test_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        section = test_df["section"].values
        scores = np.zeros(len(test_df))
        if self.sep_section:
            for sec in np.unique(section):  # type: ignore
                scores[section == sec] = self.calc_score(
                    test_df[section == sec], self.centers_dict[sec]
                )
        else:
            scores = self.calc_score(test_df, self.centers_dict[0])

        return {"plain": scores}
