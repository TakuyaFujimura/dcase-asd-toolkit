import numpy as np
from sklearn.cluster import KMeans

from .base import PseudoLabelModel


class KMeansModel(PseudoLabelModel):
    def __init__(self, num_class: int, random_state: int = 42):
        self.kmeans = KMeans(n_clusters=num_class, random_state=random_state)

    def fit_predict(self, embed: np.ndarray) -> np.ndarray:
        idx_array = self.kmeans.fit_predict(embed)
        return idx_array
