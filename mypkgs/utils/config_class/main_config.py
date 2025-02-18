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


class BasicDMConfig(BaseModel):

    class DSConfig(BaseModel):
        path_selector_list: List[str]
        use_cache: bool = False

    class CollatorConfig(BaseModel):
        sec: float
        sr: int
        shuffle: bool

        @field_validator("sr")
        def check_sr(cls, v):
            if v not in [16000]:
                raise ValueError("Unexpected sampling rate")
            return v

    dataloader: Dict[str, Any]
    dataset: DSConfig
    batch_sampler: Optional[Dict[str, Any]] = None
    collator: CollatorConfig


class BasicDMSplitConfig(BaseModel):
    train: BasicDMConfig


class MainConfig(BaseModel):
    seed: int
    name: str
    version: str
    refresh_rate: int
    num_workers: int
    tf32: bool = False
    callback_opts: Dict[str, Dict[str, Any]]
    every_n_epochs_valid: int
    result_dir: Path
    model: ModelConfig
    trainer: Dict[str, Any]
    datamodule_type: Literal["basic"]
    label_dict_path: Dict[str, Path] = {}
    datamodule: dict

    data_dir: str
    dcase: str

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
