from typing import Dict

from pydantic import BaseModel


class LabelInfo(BaseModel):
    path2idx_dict: Dict[str, int]
    num_class: int
