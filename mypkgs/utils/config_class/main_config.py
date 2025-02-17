from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, field_validator


class GradConfig(BaseModel):
    log_every_n_steps: int = 25
    clipper_cfg: Optional[Dict[str, Any]] = None


class ModelConfig(BaseModel):
    tgt_class: str
    model_cfg: Dict[str, Any]
    optim_cfg: Dict[str, Any]
    scheduler_cfg: Optional[Dict[str, Any]]
    grad_cfg: GradConfig


class BasicDMConfigSet(BaseModel):

    class DSConfig(BaseModel):
        npy_name: str
        path_list_json: Path
        use_cache: bool = False

    class CollatorConfig(BaseModel):
        sec: float
        sr: int
        shuffle: bool
        fixed_start_frame: int = 0

    dataloader: Dict[str, Any]
    dataset: DSConfig
    batch_sampler: Optional[Dict[str, Any]] = None
    collator: CollatorConfig


class BasicDMConfig(BaseModel):
    train: BasicDMConfigSet


class MainConfig(BaseModel):
    seed: int
    name: str
    version: str
    refresh_rate: int
    num_workers: int
    tf32: bool = False
    callback_opts: Dict[str, Dict[str, Any]]
    every_n_epochs_valid: int
    sampling_rate: int
    exp_root: Path
    model: ModelConfig
    trainer: Dict[str, Any]
    datamodule_type: Literal["basic"]
    label_dict_path: Dict[str, Path] = {}
    data_dir: Path
    datamodule: dict

    @field_validator("name", mode="before")
    def cast_name(cls, v):
        if isinstance(v, str):
            return v
        elif isinstance(v, int):
            return str(v)
        else:
            raise ValueError("Unexpected name type")

    @field_validator("sampling_rate")
    def check_sampling_rate(cls, v):
        if v not in [15000, 12800]:
            raise ValueError("Unexpected sampling rate")
        return v
