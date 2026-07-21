from typing import List, Optional

import numpy as np
from sklearn.cluster import KMeans

from .base import PseudoLabelModel


class KMeansModel(PseudoLabelModel):
    def __init__(
        self,
        num_class: int,
        num_class_target: Optional[int] = None,
        random_state: int = 42,
    ):
        """
        Args:
            num_class: Number of clusters.
            num_class_target: Number of clusters for the target domain. If `None`,
                clustering is performed without distinguishing between domains.
            random_state: Random seed used by KMeans.
        """
        self.num_class = num_class
        self.num_class_target = num_class_target
        self.random_state = random_state

    def _get_n_clusters(self, embed: np.ndarray, domain: str) -> int:
        if domain == "target":
            assert self.num_class_target is not None
            return self.num_class_target
        else:
            return self.num_class

    def _get_domain_array(self, embed: np.ndarray, path: List[str]) -> np.ndarray:
        if len(embed) != len(path):
            raise ValueError(
                f"embed and path lengths differ: {len(embed)} != {len(path)}"
            )

        if self.num_class_target is None:
            return np.array(["source"] * len(embed))
        else:
            domain = [p.split("/")[-1].split("_")[2] for p in path]
            assert set(domain) == {"source", "target"}
            return np.array(domain)

    def _fit_predict_by_domain(
        self, embed: np.ndarray, domain: np.ndarray
    ) -> np.ndarray:
        idx_array = np.ones(len(embed)) * -1
        offset_d = 0
        for d in np.unique(domain):
            embed_d = embed[domain == d]
            n_clusters_d = self._get_n_clusters(embed=embed_d, domain=d)
            kmeans = KMeans(n_clusters=n_clusters_d, random_state=self.random_state)
            idx_array[domain == d] = kmeans.fit_predict(embed_d) + offset_d
            offset_d += len(np.unique(idx_array[domain == d]))

        assert (idx_array != -1).all()
        return idx_array

    def fit_predict(self, embed: np.ndarray, path: List[str]) -> np.ndarray:
        domain = self._get_domain_array(embed=embed, path=path)
        return self._fit_predict_by_domain(embed=embed, domain=domain)


class KMeansRatioModel(KMeansModel):
    def __init__(
        self, ratio: float, ratio_target: Optional[float] = None, random_state: int = 42
    ):
        """
        Args:
            ratio: Ratio of clusters to samples. The number of clusters is computed as
                `max(1, int(n_samples * ratio))`.
            ratio_target: Ratio of clusters to target-domain samples. When specified,
                source and target samples are clustered separately using `ratio` and
                `ratio_target`, respectively. If `None`, clustering is performed without
                distinguishing between domains.
            random_state: Random seed used by KMeans.
        """
        if ratio_target is None:
            num_class_target = None
        else:
            num_class_target = 0

        super().__init__(
            num_class=0, num_class_target=num_class_target, random_state=random_state
        )
        self.ratio = ratio
        self.ratio_target = ratio_target

    def _get_n_clusters(self, embed: np.ndarray, domain: str) -> int:
        if domain == "target":
            assert self.ratio_target is not None
            return max(1, int(len(embed) * self.ratio_target))
        else:
            return max(1, int(len(embed) * self.ratio))
