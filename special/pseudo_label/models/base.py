from abc import ABC, abstractmethod

import numpy as np


class PseudoLabelModel(ABC):

    @abstractmethod
    def fit_predict(self, embed: np.ndarray) -> np.ndarray:
        raise NotImplementedError
