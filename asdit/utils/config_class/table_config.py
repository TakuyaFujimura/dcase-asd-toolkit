from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel, Field, field_validator

from .cast_utils import cast_str, check_dcase


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
