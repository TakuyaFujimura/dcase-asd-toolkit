import logging
from typing import Any, Dict

from asdit.models.audio_feature.stft import STFT

from ..base import BaseFrontend

logger = logging.getLogger(__name__)


class STFTPoolModel(BaseFrontend):
    def __init__(self, stft_cfg: Dict[str, Any], pool: str):
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

        self.stft = STFT(**stft_cfg)

    def extract(self, batch: dict) -> Dict[str, Any]:
        spectrogram = self.stft(batch["wave"])  # B, F, T
        if self.agg == "mean":
            embed = spectrogram.mean(dim=self.axis)
        elif self.agg == "max":
            embed = spectrogram.max(dim=self.axis).values

        return {"embed": embed}
