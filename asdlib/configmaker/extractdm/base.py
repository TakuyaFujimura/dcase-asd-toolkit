from abc import ABC, abstractmethod

from asdlib.utils.config_class import DMConfig


class BaseExtractDMConfigMaker(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_config(self, split: str) -> DMConfig:
        pass
