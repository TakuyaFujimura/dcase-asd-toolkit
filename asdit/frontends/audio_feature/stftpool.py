from typing import Any, Dict

from asdit.models.audio_feature.stft import STFT
from asdit.utils.config_class.output_config import PLOutput

from ..base import BaseFrontend


class STFTPoolModel(BaseFrontend):
    def __init__(
        self, stft_cfg: Dict[str, Any], axis: str, pool: str
    ):  # Literal["freq", "time"], pool: Literal["mean", "max"]):
        if axis == "freq":
            self.axis = 1
        elif axis == "time":
            self.axis = 2
        else:
            raise ValueError(f"Invalid axis: {axis}")

        if pool not in ["mean", "max"]:
            raise ValueError(f"Invalid pool: {pool}")
        self.pool = pool

        self.stft = STFT(**stft_cfg)

    def extract(self, batch: dict) -> PLOutput:
        spectrogram = self.stft(batch["wave"])  # B, F, T
        if self.pool == "mean":
            embed = spectrogram.mean(dim=self.axis)
        elif self.pool == "max":
            embed = spectrogram.max(dim=self.axis).values

        embed_dict = {"main": embed}
        return PLOutput(embed=embed_dict)
