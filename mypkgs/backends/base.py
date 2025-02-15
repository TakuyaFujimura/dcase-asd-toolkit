from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import pandas as pd


class BaseBackend(ABC):

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> None:
        pass

    @abstractmethod
    def anomaly_score(self, test_df: pd.DataFrame) -> Dict[str, np.ndarray]:
        pass

    def check_target(self, is_target: np.ndarray) -> None:
        if np.sum(is_target) == 0:
            raise ValueError("number of target data is 0.")
