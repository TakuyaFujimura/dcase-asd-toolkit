from typing import Any, Dict

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.cluster import KMeans

from .base import BaseBackend
from .utils import get_embed_from_df


def min_euclid(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Args:
        x (np.ndarray): (N, D)
        y (np.ndarray): (M, D)

    Returns:
        np.ndarray : minimum euclidean distance from x to y
    """
    x_squared_sum = np.sum(x**2, axis=-1)[:, np.newaxis]  # (N, 1)
    y_squared_sum = np.sum(y**2, axis=-1)[np.newaxis, :]  # (1, M)
    inner_product = np.dot(x, y.transpose())  # (N, M)
    distances_squared = x_squared_sum + y_squared_sum - 2 * inner_product
    distances = np.sqrt(distances_squared)  # (N, M)
    return np.min(distances, axis=-1)  # (N,)


class KmeansConfig(BaseModel):
    n_clusters: int


class Kmeans(BaseBackend):
    def __init__(self, hp_dict: Dict[str, Any]):
        self.hp_dict = KmeansConfig(**hp_dict)
        self.kmeans_so = KMeans(n_clusters=self.hp_dict.n_clusters, random_state=0)

    def fit(self, train_df: pd.DataFrame):
        is_target = np.asarray(train_df["is_target"].values)
        self.check_target(is_target)
        embed = get_embed_from_df(train_df)
        self.kmeans_so.fit(embed[is_target == 0])
        self.centers_ta: np.ndarray = embed[is_target == 1]  # (M, D)

    def anomaly_score(self, test_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        embed = get_embed_from_df(test_df)  # (N, D)
        centers_so = self.kmeans_so.cluster_centers_
        centers = np.vstack([centers_so, self.centers_ta])  # (M, D)
        scores = min_euclid(embed, centers)  # (N,)
        return {"plain": scores}
