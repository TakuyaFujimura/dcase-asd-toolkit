from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from .cast_utils import cast_str, check_dcase


class ExtractDMConfig(BaseModel):
    dataloader: Dict[str, Any]
    dataset: Dict[str, Any]
    batch_sampler: Optional[Dict[str, Any]] = None
    collator: Dict[str, Any]


class MainExtractConfig(BaseModel):
    device: str
    seed: int
    dcase: str
    name: str
    version: str

    frontend_source_type: Literal["checkpoint", "config"]

    frontend_config: Optional[Dict[str, Any]] = None
    frontend_checkpoint: Optional[str] = None
    model_ver: Optional[str] = None

    result_dir: Path
    infer_ver: str

    data_dir: str
    datamodule: Dict[str, ExtractDMConfig] = Field(min_length=1)

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
