from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from .cast_utils import cast_str, check_dcase
from .train_config import DMConfig


class ExtractDMSplitConfig(BaseModel):
    train: DMConfig
    test: DMConfig


class MainExtractConfig(BaseModel):
    device: str
    seed: int
    dcase: str
    name: str
    version: str

    restore_or_scratch: Literal["restore", "scratch"]

    scratch_frontend: Optional[Dict[str, Any]] = None
    restore_model_ver: Optional[str] = None
    restore_ckpt_ver: Optional[str] = None

    result_dir: Path
    infer_ver: str

    data_dir: str
    datamodule: ExtractDMSplitConfig

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
