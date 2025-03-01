from pathlib import Path
from typing import List

from pydantic import BaseModel, Field, field_validator


class MainTableConfig(BaseModel):
    seed: int
    name: str
    version: str

    ckpt_ver: str
    result_dir: Path

    additional_metric_list: List[str] = Field(default_factory=list)
    dcase: str
    overwrite: bool = False

    @field_validator("name", mode="before")
    def cast_name(cls, v):
        if isinstance(v, str):
            return v
        elif isinstance(v, int):
            return str(v)
        else:
            raise ValueError("Unexpected name type")
