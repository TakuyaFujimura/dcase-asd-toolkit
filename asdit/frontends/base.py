from abc import ABC, abstractmethod

from asdit.utils.config_class import FrontendOutput


class BaseFrontend(ABC):
    @abstractmethod
    def extract(self, batch: dict) -> FrontendOutput:
        pass
