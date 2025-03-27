from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


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
        if isinstance(v, str):
            return v
        elif isinstance(v, int):
            return str(v)
        else:
            raise ValueError("Unexpected name type")
