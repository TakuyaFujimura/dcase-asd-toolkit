from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseFrontend(ABC):
    @abstractmethod
    def extract(self, batch: dict) -> Dict[str, Any]:
        pass
