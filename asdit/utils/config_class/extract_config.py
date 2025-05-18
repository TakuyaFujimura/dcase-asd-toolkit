from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from .cast_utils import cast_str, check_dcase


class MainExtractConfig(BaseModel):
    device: str
    seed: int
    dcase: str
    name: str
    version: str

    resume_or_scratch: Literal["resume", "scratch"]

    frontend_cfg: Optional[Dict[str, Any]] = None
    model_ver: Optional[str] = None
    ckpt_ver: Optional[str] = None

    result_dir: Path
    infer_ver: str

    datamodule: dict
    label_dict_path: Dict[str, Path] = Field(default_factory=dict)

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
