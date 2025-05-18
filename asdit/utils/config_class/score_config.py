from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, Field, field_validator

from .cast_utils import cast_str, check_dcase


class MainScoreConfig(BaseModel):
    seed: int
    dcase: str
    name: str
    version: str

    infer_ver: str
    result_dir: Path

    backend: List[Dict[str, Any]] = Field(default_factory=list)

    machine: str

    extract_items: List[str] = Field(default_factory=list)

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
