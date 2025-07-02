import logging
from abc import ABC, abstractmethod
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)


class BaseBackend(ABC):

    @abstractmethod
    def fit(self, train_dict: dict) -> None:
        pass

    @abstractmethod
    def anomaly_score(self, test_dict: dict) -> Dict[str, np.ndarray]:
        pass

    def check_target(self, is_target: np.ndarray) -> None:
        if np.sum(is_target) == 0:
            logger.warning("number of target data is 0.")
