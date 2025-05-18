from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel, Field, field_validator

from .cast_utils import cast_str, check_dcase


def check_hmean_cfg_dict(v: Dict[str, List[str]]) -> Dict[str, List[str]]:
    for name, metric_list in v.items():
        if name.startswith("official"):
            raise ValueError(
                "name starting with 'official' is reserved. Please rename it."
            )
        for metric in metric_list:
            if metric[0] in ["0", "1", "2", "3", "4", "5", "6", "7"]:
                raise ValueError(
                    f"{name}: hmean_cfg_dict cannot specify section.\n"
                    + "Please remove it (e.g. '0_s_auc' -> 's_auc').\n"
                    + "This is because the section to be collected is automatically determined by dcase."
                )
    return v


class MainEvaluateConfig(BaseModel):
    seed: int
    dcase: str
    name: str
    version: str

    infer_ver: str
    result_dir: Path

    hmean_cfg_dict: Dict[str, List[str]] = Field(default_factory=dict)

    machine: str

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

    @field_validator("hmean_cfg_dict", mode="after")
    def check_hmean_cfg_dict_(cls, v: Dict[str, List[str]]):
        return check_hmean_cfg_dict(v)
