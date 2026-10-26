from pathlib import Path
from typing import List, Optional

import numpy as np
from sklearn.cluster import KMeans

from .base import PseudoLabelModel


class KMeansModel(PseudoLabelModel):
    def __init__(
        self,
        num_class: int,
        num_class_target: Optional[int] = None,
        split_by_domain: bool = False,
        random_state: int = 42,
    ):
        """
        Args:
            num_class: Number of clusters for all samples or the source domain.
            num_class_target: Number of clusters for the target domain. If `None`,
                uses `num_class`. Requires `split_by_domain=True`.
            split_by_domain: Cluster source and target domains independently.
            random_state: Random seed used by KMeans.
        """
        if num_class_target is not None:
            if not split_by_domain:
                raise ValueError("num_class_target requires split_by_domain=True")
            if num_class_target <= 0:
                raise ValueError("num_class_target must be positive")
        self.num_class = num_class
        self.num_class_target = num_class_target
        self.split_by_domain = split_by_domain
        self.random_state = random_state

    def _get_n_clusters(self, embed: np.ndarray, domain: str) -> int:
        if domain == "target" and self.num_class_target is not None:
            return self.num_class_target
        else:
            return self.num_class

    def _get_domain_array(self, embed: np.ndarray, path: List[str]) -> np.ndarray:
        if len(embed) != len(path):
            raise ValueError(
                f"embed and path lengths differ: {len(embed)} != {len(path)}"
            )

        if not self.split_by_domain:
            return np.array(["source"] * len(embed))
        else:
            domain = [Path(p).name.split("_")[2] for p in path]
            if not set(domain).issubset({"source", "target"}):
                raise ValueError("Expected source/target domain in DCASE filenames")
            return np.array(domain)

    def _fit_predict_by_domain(
        self, embed: np.ndarray, domain: np.ndarray
    ) -> np.ndarray:
        idx_array = np.full(len(embed), -1, dtype=np.int64)
        offset_d = 0
        for d in np.unique(domain):
            embed_d = embed[domain == d]
            n_clusters_d = self._get_n_clusters(embed=embed_d, domain=d)
            if not 1 <= n_clusters_d <= len(embed_d):
                raise ValueError(
                    f"Expected 1 <= n_clusters <= {len(embed_d)} for {d}, "
                    f"got {n_clusters_d}"
                )
            kmeans = KMeans(n_clusters=n_clusters_d, random_state=self.random_state)
            # Duplicate embeddings may leave clusters empty. Compact IDs before
            # offsetting so classes from different domains/machines never overlap.
            _, labels = np.unique(kmeans.fit_predict(embed_d), return_inverse=True)
            idx_array[domain == d] = labels + offset_d
            offset_d += len(np.unique(labels))

        assert (idx_array != -1).all()
        return idx_array

    def fit_predict(self, embed: np.ndarray, path: List[str]) -> np.ndarray:
        if embed.ndim != 2 or len(embed) == 0:
            raise ValueError("embed must be a nonempty 2-D array")
        domain = self._get_domain_array(embed=embed, path=path)
        return self._fit_predict_by_domain(embed=embed, domain=domain)


class KMeansRatioModel(KMeansModel):
    def __init__(
        self,
        ratio: float,
        ratio_target: Optional[float] = None,
        num_class_target: Optional[int] = None,
        split_by_domain: bool = False,
        random_state: int = 42,
    ):
        """
        Args:
            ratio: Ratio of clusters to samples. The number of clusters is computed as
                `max(1, int(n_samples * ratio))`.
            ratio_target: Ratio for target-domain samples. If neither target setting
                is specified, uses `ratio` on each domain's sample count.
            num_class_target: Fixed target cluster count instead of `ratio_target`.
                Both target settings require `split_by_domain=True`.
            split_by_domain: Cluster source and target domains independently.
            random_state: Random seed used by KMeans.
        """
        if not 0 < ratio <= 1:
            raise ValueError("ratio must be in (0, 1]")
        if ratio_target is not None and not 0 < ratio_target <= 1:
            raise ValueError("ratio_target must be in (0, 1]")
        if ratio_target is not None:
            if not split_by_domain:
                raise ValueError("ratio_target requires split_by_domain=True")
            if num_class_target is not None:
                raise ValueError(
                    "ratio_target and num_class_target are mutually exclusive"
                )

        super().__init__(
            num_class=1,
            num_class_target=num_class_target,
            split_by_domain=split_by_domain,
            random_state=random_state,
        )
        self.ratio = ratio
        self.ratio_target = ratio_target

    def _get_n_clusters(self, embed: np.ndarray, domain: str) -> int:
        ratio = self.ratio
        if domain == "target":
            if self.num_class_target is not None:
                return self.num_class_target
            if self.ratio_target is not None:
                ratio = self.ratio_target
        return max(1, int(len(embed) * ratio))
