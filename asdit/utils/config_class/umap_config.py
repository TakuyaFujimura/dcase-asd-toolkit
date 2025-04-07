from pathlib import Path
from typing import List

from pydantic import BaseModel, Field, field_validator


class MainUmapConfig(BaseModel):
    seed: int
    dcase: str
    name: str
    version: str

    infer_ver: str
    result_dir: Path

    machine: str

    metric: str
    embed_key: str

    extract_items: List[str] = Field(default_factory=list)

    overwrite: bool

    @field_validator("name", mode="before")
    def cast_name(cls, v):
        if isinstance(v, str):
            return v
        elif isinstance(v, (int, float)):
            return str(v)
        else:
            raise ValueError("Unexpected name type")

    @field_validator("version", mode="before")
    def cast_version(cls, v):
        if isinstance(v, str):
            return v
        elif isinstance(v, (int, float)):
            return str(v)
        else:
            raise ValueError("Unexpected version type")
