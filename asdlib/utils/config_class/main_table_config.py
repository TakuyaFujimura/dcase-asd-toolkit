from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from .main_evaluate_config import HmeanCfgDict


class MainTableConfig(BaseModel):
    seed: int
    name: str
    version: str

    ckpt_ver: str
    result_dir: Path

    hmean_cfg_dict: HmeanCfgDict = Field(default_factory=dict)
    dcase: str
    overwrite: bool = False

    @field_validator("name", mode="before")
    def cast_name(cls, v):
        if isinstance(v, str):
            return v
        elif isinstance(v, int):
            return str(v)
        else:
            raise ValueError("Unexpected name type")

    @field_validator("hmean_cfg_dict", mode="after")
    def check_hmean_cfg_dict(cls, v: HmeanCfgDict):
        for name, metric_list in v.items():
            for metric in metric_list:
                if metric[0] in ["0", "1", "2", "3", "4", "5"]:
                    raise ValueError(
                        f"{name}: hmean_cfg_dict cannot specify section.\n"
                        + "Please remove it (e.g. '0_s_auc' -> 's_auc').\n"
                        + "This is because the section to be collected is automatically determined by dcase."
                    )
        return v
