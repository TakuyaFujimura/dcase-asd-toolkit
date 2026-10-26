from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, Optional

import numpy as np


class BaseBackend(ABC):
    """Fit reference features and return one anomaly score per query sample.

    Sample arrays and JSON metadata are separate inputs. Scores preserve query
    order; larger values denote more anomalous samples.
    """

    @abstractmethod
    def fit(
        self, train_dict: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Fit using sample arrays and optional frontend output metadata."""
        pass

    @abstractmethod
    def anomaly_score(
        self,
        extract_dict: Dict[str, Any],
        split: Literal["train", "test"] = "test",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Args:
            extract_dict: dictionary containing the test data information
            metadata: frontend output metadata, separate from sample arrays
            split: indicates whether to score the training data or the test data.
                For example, KNN methods exclude self embeddings for "train" split.

        Returns:
            Score names mapped to arrays shaped (B,), in query order.
        """

        pass


class EmbeddingBackend(BaseBackend):
    """Backend that consumes one embedding array under a public embed_key.

    TransformWrapper uses this contract to route each transformed view to its
    child.
    """

    embed_key: str
