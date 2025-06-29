import logging
from typing import Dict

import numpy as np
from sklearn.cluster import KMeans

from asdit.utils.common.np_util import normalize_vector

from .base import BaseBackend

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
    def __init__(
        self,
        n_clusters: int,
        metric: str = "cosine",
        sep_section: bool = False,
        embed_key="embed",
    ):
        """
        Args:
            n_clusters (int): n_clusters for KMeans
            metric (str): Distance metric. Options are "euclid" or "cosine". Defaults to "cosine".
            sep_section (bool, optional): Whether to separately construct a backend for each section. Defaults to False.
            embed_key (str, optional): Key to access embeddings in the extract_dict. Defaults to "embed".
        """
        if metric not in ["euclid", "cosine"]:
            raise ValueError(f"Unexpected metric: {metric}")
        self.metric = metric
        self.n_clusters = n_clusters
        self.sep_section = sep_section
        self.centers_dict: Dict[int, np.ndarray] = {}
        self.embed_key = embed_key

    def get_section(self, extract_dict: dict) -> np.ndarray:
        section = extract_dict["section"]
        if not self.sep_section:
            section = np.zeros_like(section)
        return section

    def get_centers(self, embed: np.ndarray, is_target: np.ndarray) -> np.ndarray:
        """
        Args:
            embed (np.ndarray): (N, D)
            is_target (np.ndarray): (N,)

        Returns:
            centers (np.ndarray): (M, D)
        """
        self.check_target(is_target)

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

    def fit(self, train_dict: dict):
        section = self.get_section(train_dict)
        is_target = np.asarray(train_dict["is_target"])
        embed = train_dict[self.embed_key]

        for sec in np.unique(section):  # type: ignore
            idx = section == sec
            self.centers_dict[sec] = self.get_centers(
                embed=embed[idx], is_target=is_target[idx]
            )

    def calc_score(self, embed: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """
        Args:
            embed (np.ndarray): (N, D)
            centers (np.ndarray): (M, D)

        Returns:
            scores (np.ndarray): (N,)
        """
        if self.metric == "cosine":
            scores = min_cosine(embed, centers)  # (N,)
        else:
            scores = min_euclid(embed, centers)  # (N,)
        return scores

    def anomaly_score(self, test_dict: dict) -> Dict[str, np.ndarray]:
        section = self.get_section(test_dict)
        embed = test_dict[self.embed_key]
        scores = np.zeros(len(section))

        for sec in np.unique(section):  # type: ignore
            idx = section == sec
            scores[idx] = self.calc_score(embed[idx], self.centers_dict[sec])

        return {"main": scores}
