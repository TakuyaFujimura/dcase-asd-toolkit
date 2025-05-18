from abc import ABC, abstractmethod

from asdit.utils.config_class import PLOutput


class BaseFrontend(ABC):
    @abstractmethod
    def extract(self, batch: dict) -> PLOutput:
        pass
