from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator

from .datamodule import AudioFeatDSConfig, BasicCollatorConfig, BasicDSConfig


class GradConfig(BaseModel):
    log_every_n_steps: int = 25
    clipper_cfg: Optional[Dict[str, Any]] = None


class ModelConfig(BaseModel):
    tgt_class: str
    model_cfg: Dict[str, Any]
    optim_cfg: Dict[str, Any]
    scheduler_cfg: Optional[Dict[str, Any]]
    grad_cfg: GradConfig


class DMConfig(BaseModel):
    dataloader: Dict[str, Any]
    dataset: BasicDSConfig | AudioFeatDSConfig
    batch_sampler: Optional[Dict[str, Any]] = None
    collator: BasicCollatorConfig


class DMSplitConfig(BaseModel):
    train: DMConfig
    valid: Optional[DMConfig] = None


class MainTrainConfig(BaseModel):
    seed: int
    name: str
    version: str
    refresh_rate: int
    num_workers: int
    double_precision: bool = False
    callback_opts: Dict[str, Dict[str, Any]]
    every_n_epochs_valid: int
    result_dir: Path
    model: ModelConfig
    trainer: Dict[str, Any]
    label_dict_path: Dict[str, Path] = Field(default_factory=dict)
    datamodule: DMSplitConfig

    data_dir: str
    dcase: str
    model_ver: str

    @field_validator("name", mode="before")
    def cast_name(cls, v):
        if isinstance(v, str):
            return v
        elif isinstance(v, int):
            return str(v)
        else:
            raise ValueError("Unexpected name type")

    @field_validator("dcase", mode="before")
    def check_dcase(cls, v):
        if v in [f"dcase202{i}" for i in range(5)]:
            return v
        else:
            raise ValueError("Unexpected dcase type")
