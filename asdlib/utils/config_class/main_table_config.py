from pathlib import Path

from pydantic import BaseModel, field_validator


class MainTableConfig(BaseModel):
    seed: int
    name: str
    version: str

    ckpt_ver: str
    result_dir: Path

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
