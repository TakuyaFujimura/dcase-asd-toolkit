from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, Field, field_validator


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
        if isinstance(v, str):
            return v
        elif isinstance(v, int):
            return str(v)
        else:
            raise ValueError("Unexpected name type")
