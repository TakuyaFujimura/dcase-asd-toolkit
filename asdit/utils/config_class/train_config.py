from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator

from .cast_utils import cast_str, check_dcase


class DMConfig(BaseModel):
    dataloader: Dict[str, Any]
    dataset: Dict[str, Any]
    batch_sampler: Optional[Dict[str, Any]] = None
    collator: Dict[str, Any]


class DMSplitConfig(BaseModel):
    train: DMConfig
    valid: Optional[DMConfig] = None


class MainTrainConfig(BaseModel):
    seed: int
    dcase: str
    name: str
    version: str
    result_dir: Path
    data_dir: str

    frontend: Dict[str, Any]
    trainer: Dict[str, Any]
    datamodule: DMSplitConfig

    model_ver: str

    refresh_rate: int = 1
    callback: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    resume_ckpt_path: Optional[str] = None

    @field_validator("name", mode="before")
    def cast_name(cls, v):
        return cast_str(v)

    @field_validator("version", mode="before")
    def cast_version(cls, v):
        return cast_str(v)

    @field_validator("dcase", mode="before")
    def check_dcase(cls, v):
        return check_dcase(v)
