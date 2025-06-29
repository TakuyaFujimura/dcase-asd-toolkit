from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator

from .cast_utils import cast_str, check_dcase


class GradConfig(BaseModel):
    log_every_n_steps: int = 25
    clipper_cfg: Optional[Dict[str, Any]] = None


class FrontendConfig(BaseModel):
    tgt_class: str
    model_cfg: Dict[str, Any]
    optim_cfg: Dict[str, Any]
    lrscheduler_cfg: Optional[Dict[str, Any]] = None
    grad_cfg: GradConfig


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

    frontend: FrontendConfig
    trainer: Dict[str, Any]
    label_dict_path: Dict[str, Path] = Field(default_factory=dict)
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
