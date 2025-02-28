from pathlib import Path
from typing import List

from pydantic import BaseModel, Field, field_validator


class CustomHmeanConfig(BaseModel):
    name: str
    cols: List[str]


class MainEvaluateConfig(BaseModel):
    seed: int
    name: str
    version: str

    ckpt_ver: str
    result_dir: Path

    hmean_cfg_list: List[str | CustomHmeanConfig] = Field(default_factory=list)

    machine: str

    overwrite: bool = False

    @field_validator("name", mode="before")
    def cast_name(cls, v):
        if isinstance(v, str):
            return v
        elif isinstance(v, int):
            return str(v)
        else:
            raise ValueError("Unexpected name type")
