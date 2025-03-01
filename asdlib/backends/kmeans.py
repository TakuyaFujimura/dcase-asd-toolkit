from typing import Any, Dict, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.cluster import KMeans

from asdlib.utils.common import get_embed_from_df

from .base import BaseBackend
from .utils import normalize_vector


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


class KmeansConfig(BaseModel):
    n_clusters: int
    metric: Literal["euclid", "cosine"]


class Kmeans(BaseBackend):
    def __init__(self, hp_dict: Dict[str, Any]):
        self.hp_dict = KmeansConfig(**hp_dict)
        self.kmeans_so = KMeans(n_clusters=self.hp_dict.n_clusters, random_state=0)

    def fit(self, train_df: pd.DataFrame):
        is_target = np.asarray(train_df["is_target"].values)
        self.check_target(is_target)
        embed = get_embed_from_df(train_df)

        if self.hp_dict.metric == "cosine":
            embed = normalize_vector(embed)

        centers_ta: np.ndarray = embed[is_target == 1]  # (M, D)
        self.kmeans_so.fit(embed[is_target == 0])
        centers_so: np.ndarray = self.kmeans_so.cluster_centers_

        if self.hp_dict.metric == "cosine":
            centers_so = normalize_vector(centers_so)

        self.centers = np.vstack([centers_so, centers_ta])  # (M, D)

    def anomaly_score(self, test_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        embed = get_embed_from_df(test_df)  # (N, D)

        if self.hp_dict.metric == "cosine":
            scores = min_cosine(embed, self.centers)  # (N,)
        else:
            scores = min_euclid(embed, self.centers)  # (N,)
        return {"plain": scores}
