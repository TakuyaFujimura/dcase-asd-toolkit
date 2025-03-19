from typing import List, Literal

from pydantic import BaseModel, field_validator


class BasicDSConfig(BaseModel):
    tgt_class: Literal["asdit.datasets.BasicDataset"]
    path_selector_list: List[str]
    use_cache: bool = False


class BasicCollatorConfig(BaseModel):
    tgt_class: Literal["asdit.datasets.BasicCollator"]
    sec: float | Literal["all", "none"]
    sr: int
    need_feat: bool = False
    shuffle: bool

    @field_validator("sr")
    def check_sr(cls, v):
        if v not in [16000]:
            raise ValueError("Unexpected sampling rate")
        return v


class AudioFeatDSConfig(BaseModel):
    tgt_class: Literal["asdit.datasets.AudioFeatDataset"]
    path_selector_list: List[str]
    audio_feat_cfg: dict
