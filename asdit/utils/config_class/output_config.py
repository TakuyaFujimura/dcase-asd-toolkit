from typing import Dict

from pydantic import BaseModel, Field

from .tensor_class import TorchTensor


class PLOutput(BaseModel):
    embed: Dict[str, TorchTensor] = Field(default_factory=dict)
    logits: Dict[str, TorchTensor] = Field(default_factory=dict)
    AS: Dict[str, TorchTensor] = Field(default_factory=dict)
