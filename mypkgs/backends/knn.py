from typing import Any, Dict

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from pydantic import BaseModel
from sklearn.neighbors import NearestNeighbors

from .base import BaseBackend
from .utils import get_embed_from_df


class KnnConfig(BaseModel):
    n_neighbors_so: int
    n_neighbors_ta: int
    smote_ratio: float = 0
    smote_neighbors: int = 5


class Knn(BaseBackend):
    def __init__(self, hp_dict: Dict[str, Any]):
        self.hp_dict = KnnConfig(**hp_dict)
        self.knn_so = NearestNeighbors(
            n_neighbors=self.hp_dict.n_neighbors_so, metric="euclidean"
        )
        self.knn_ta = NearestNeighbors(
            n_neighbors=self.hp_dict.n_neighbors_ta, metric="euclidean"
        )
        if self.hp_dict.smote_ratio == 0:
            self.smote = None
        else:
            self.smote = SMOTE(
                sampling_strategy=self.hp_dict.smote_ratio,  # type: ignore
                random_state=0,
                k_neighbors=self.hp_dict.smote_neighbors,
            )

    def fit(self, train_df: pd.DataFrame) -> None:
        is_target = np.asarray(train_df["is_target"].values)
        self.check_target(is_target)
        embed = get_embed_from_df(train_df)
        if self.smote is not None:
            embed, is_target = self.smote.fit_resample(embed, is_target)  # type: ignore
        self.knn_so.fit(embed[is_target == 0])  # type: ignore
        self.knn_ta.fit(embed[is_target == 1])  # type: ignore

    def anomaly_score(self, test_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        embed = get_embed_from_df(test_df)
        scores_so: np.ndarray = self.knn_so.kneighbors(embed)[0].mean(1)
        scores_ta: np.ndarray = self.knn_ta.kneighbors(embed)[0].mean(1)
        return {"plain": np.minimum(scores_so, scores_ta)}
