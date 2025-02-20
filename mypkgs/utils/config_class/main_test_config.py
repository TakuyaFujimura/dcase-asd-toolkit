from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, field_validator


class UmapConfig(BaseModel):
    metric: str
    vis_type: str = "standard"
    overwrite: bool = True


class EvaluateConfig(BaseModel):
    hmean_list: List[str] = []


class MainTestConfig(BaseModel):
    extract: bool = True
    score: bool = True
    evaluate: bool = True
    umap: bool = True
    table_metric_list: List[str] = []

    device: str
    seed: int
    name: str
    version: str
    infer_ver: str
    result_dir: Path
    tf32: bool = False

    batch_size: int
    num_workers: int
    backend: List[Dict[str, Any]]

    path_selector_list: Optional[List[str]] = None
    evaluate_cfg: Optional[EvaluateConfig] = None
    umap_cfg: Optional[UmapConfig] = None

    @field_validator("name", mode="before")
    def cast_name(cls, v):
        if isinstance(v, str):
            return v
        elif isinstance(v, int):
            return str(v)
        else:
            raise ValueError("Unexpected name type")
