from abc import ABC, abstractmethod

from asdit.utils.config_class import PLOutput


class BaseASDModel(ABC):
    @abstractmethod
    def forward(self, batch: dict) -> PLOutput:
        pass
