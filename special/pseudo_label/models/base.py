from abc import ABC, abstractmethod
from typing import List

import numpy as np


class PseudoLabelModel(ABC):

    @abstractmethod
    def fit_predict(self, embed: np.ndarray, path: List[str]) -> np.ndarray:
        raise NotImplementedError
