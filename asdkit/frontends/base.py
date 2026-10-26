from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseFrontend(ABC):
    """Extraction interface; output keys also identify entries in metadata."""

    def get_metadata(self) -> Dict[str, Any]:
        """
        Returns:
            JSON-serializable metadata.
        """
        return {}

    @abstractmethod
    def extract(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            batch: Waveforms at "wave" with shape (B, samples), plus metadata.

        Returns:
            Embeddings or anomaly scores, indexed by output name.
            Each output has B samples along its first dimension.
        """
        pass
