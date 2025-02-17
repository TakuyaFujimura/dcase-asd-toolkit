from pathlib import Path
from typing import Any, Dict, List

from pydantic import BaseModel, field_validator


class MainTestConfig(BaseModel):
    extract: bool = True
    score: bool = True
    evaluate: bool = True

    device: str
    seed: int
    name: str
    version: str
    infer_ver: str
    result_dir: Path
    tf32: bool = False

    batch_size: int
    num_workers: int
    backend: List[Dict[str, Any]]

    @field_validator("name", mode="before")
    def cast_name(cls, v):
        if isinstance(v, str):
            return v
        elif isinstance(v, int):
            return str(v)
        else:
            raise ValueError("Unexpected name type")
