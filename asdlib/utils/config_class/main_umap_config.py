from pathlib import Path
from typing import List

from pydantic import BaseModel, Field, field_validator


class MainUmapConfig(BaseModel):
    seed: int
    name: str
    version: str

    ckpt_ver: str
    result_dir: Path

    machine: str
    
    metric: str
    vis_type: str
    embed_key: str

    trans_exec: bool
    trans_overwrite: bool
    vis_exec: bool
    vis_overwrite: bool
    

    @field_validator("name", mode="before")
    def cast_name(cls, v):
        if isinstance(v, str):
            return v
        elif isinstance(v, int):
            return str(v)
        else:
            raise ValueError("Unexpected name type")
