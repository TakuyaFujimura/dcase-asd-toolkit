from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from .cast_utils import cast_str, check_dcase


class DMConfig(BaseModel):
    dataloader: Dict[str, Any]
    dataset: Dict[str, Any]
    batch_sampler: Optional[Dict[str, Any]] = None
    collator: Dict[str, Any]


class DMSplitConfig(BaseModel):
    train: DMConfig
    valid: Optional[DMConfig] = None


class CallbackConfig(BaseModel):
    tqdm_refresh_rate: int = 1
    callbacks: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


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
    callback: CallbackConfig
    other_callbacks: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    resume_ckpt_path: Optional[str] = None
    checkpoint_weights_only: bool = True

    @field_validator("name", mode="before")
    def cast_name(cls, v):
        return cast_str(v)

    @field_validator("version", mode="before")
    def cast_version(cls, v):
        return cast_str(v)

    @field_validator("dcase", mode="before")
    def check_dcase(cls, v):
        return check_dcase(v)

    @model_validator(mode="after")
    def check_dynamic_loss_with_ddp(self) -> "MainTrainConfig":
        if self._is_ddp_strategy(
            self.trainer.get("strategy")
        ) and self._is_dynamic_loss(self.frontend):
            raise ValueError(
                "frontend.model_cfg.loss_cfg.dynamic=True is not supported with DDP "
                "because dynamic loss state is updated independently on each rank. "
                "Set dynamic=False or use a non-DDP trainer strategy."
            )
        return self

    @staticmethod
    def _is_ddp_strategy(strategy: Any) -> bool:
        if isinstance(strategy, str):
            return strategy.lower().startswith("ddp")
        return False

    @staticmethod
    def _is_dynamic_loss(frontend: Dict[str, Any]) -> bool:
        model_cfg = frontend.get("model_cfg", {})
        if not isinstance(model_cfg, dict):
            return False
        loss_cfg = model_cfg.get("loss_cfg", {})
        if not isinstance(loss_cfg, dict):
            return False
        dynamic = loss_cfg.get("dynamic", True)
        if isinstance(dynamic, str):
            return dynamic.lower() == "true"
        else:
            assert isinstance(dynamic, bool)
            return dynamic
