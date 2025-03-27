from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel, Field, field_validator

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
        if isinstance(v, str):
            return v
        elif isinstance(v, int):
            return str(v)
        else:
            raise ValueError("Unexpected name type")

    @field_validator("hmean_cfg_dict", mode="after")
    def check_hmean_cfg_dict_(cls, v: Dict[str, List[str]]):
        return check_hmean_cfg_dict(v)
