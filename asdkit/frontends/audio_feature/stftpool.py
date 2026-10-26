import logging
from typing import Any, Dict

from torch import nn

from asdkit.models.audio_feature.stft import Spectrogram

from ..base import BaseFrontend

logger = logging.getLogger(__name__)


class STFTPoolFrontend(nn.Module, BaseFrontend):
    def __init__(self, stft_cfg: Dict[str, Any], pool: str):
        """
        Args:
            stft_cfg (Dict[str, Any]): Configuration for the STFT.
            pool (str): Pooling method in the form of '<axis>_<agg>', where <axis> is 'freq' or 'time' and <agg> is 'mean' or 'max'.
        """
        super().__init__()
        if len(pool.split("_")) != 2:
            raise ValueError(
                f"Invalid pool: {pool}. Must be in the form of '<axis>_<pool>'"
            )
        axis, agg = pool.split("_")
        if axis == "freq":
            self.axis = 1
        elif axis == "time":
            self.axis = 2
        else:
            raise ValueError(f"Invalid axis: {axis}")

        if agg not in ["mean", "max"]:
            raise ValueError(f"Invalid agg: {agg}")
        self.agg = agg

        self.stft = Spectrogram(**stft_cfg)

    def get_metadata(self) -> Dict[str, Any]:
        return {"embeddings": {"embed": {"representation": "vector"}}}

    def extract(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        spectrogram = self.stft(batch["wave"])  # B, F, T
        if self.agg == "mean":
            embed = spectrogram.mean(dim=self.axis)
        elif self.agg == "max":
            embed = spectrogram.max(dim=self.axis).values

        return {"embed": embed}
