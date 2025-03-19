from abc import ABC, abstractmethod
from typing import List


class LabelerBase(ABC):

    @abstractmethod
    def fit(self, all_path_list: List[str]) -> None:
        pass

    @abstractmethod
    def trans(self, path: str) -> int:
        pass
