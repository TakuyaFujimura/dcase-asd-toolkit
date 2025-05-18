from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel, Field, field_validator

from .cast_utils import cast_str, check_dcase
from .evaluate_config import check_hmean_cfg_dict


class MainTableConfig(BaseModel):
    seed: int
    dcase: str
    name: str
    version: str

    infer_ver: str
    result_dir: Path

    hmean_cfg_dict: Dict[str, List[str]] = Field(default_factory=dict)
    overwrite: bool = False

    @field_validator("name", mode="before")
    def cast_name(cls, v):
        return cast_str(v)

    @field_validator("version", mode="before")
    def cast_version(cls, v):
        return cast_str(v)

    @field_validator("dcase", mode="before")
    def check_dcase(cls, v):
        return check_dcase(v)

    @field_validator("hmean_cfg_dict", mode="after")
    def check_hmean_cfg_dict_(cls, v: Dict[str, List[str]]):
        return check_hmean_cfg_dict(v)
