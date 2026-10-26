"""Apply embedding views to selected outputs of a configured or restored frontend."""

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

from torch import nn

from asdkit.models.pretrained_models.views import build_embedding_views
from asdkit.utils.common import instantiate

from .base import BaseFrontend


@dataclass(frozen=True)
class EmbeddingViewConfig:
    """Source embedding and explicit options for building one saved view."""

    source_key: str
    view_name: str
    pooling_name: str = "mean"


class EmbeddingViewWrapperFrontend(nn.Module, BaseFrontend):
    """Build named views while preserving outputs not replaced by a saved name.

    Each postprocessing key is a saved output name; source_key selects its input.
    All views read the original outputs, even when a saved name replaces an
    input key. The wrapped frontend must identify the raw
    embedding representation in its metadata. Checkpoints are restored on CPU;
    the extraction pipeline moves the entire wrapper to the requested device.
    """

    def __init__(
        self,
        frontend_source_type: Literal["config", "checkpoint"],
        postprocessing: Dict[str, Union[EmbeddingViewConfig, Dict[str, str]]],
        frontend_config: Optional[Dict[str, Any]] = None,
        frontend_checkpoint: Optional[Union[str, Path]] = None,
    ) -> None:
        super().__init__()
        if frontend_source_type == "config":
            if frontend_config is None or frontend_checkpoint is not None:
                raise ValueError("config source requires only frontend_config")
            frontend = instantiate(frontend_config)
        elif frontend_source_type == "checkpoint":
            if frontend_checkpoint is None or frontend_config is not None:
                raise ValueError("checkpoint source requires only frontend_checkpoint")
            # Lazy import avoids the frontend registry / restore import cycle.
            from asdkit.utils.asdkit_utils.restore import restore_plfrontend

            frontend = restore_plfrontend(ckpt_path=frontend_checkpoint)
        else:
            raise ValueError(f"Unknown frontend_source_type: {frontend_source_type}")
        if not isinstance(frontend, BaseFrontend):
            raise TypeError("Wrapped frontend must implement BaseFrontend")
        self.frontend = frontend
        self.postprocessing = {
            name: (
                options
                if isinstance(options, EmbeddingViewConfig)
                else EmbeddingViewConfig(**options)
            )
            for name, options in postprocessing.items()
        }
        self.source_metadata = deepcopy(frontend.get_metadata())
        self.metadata = deepcopy(self.source_metadata)
        for name, options in self.postprocessing.items():
            info = self.source_metadata.get("embeddings", {}).get(
                options.source_key, {}
            )
            if not info.get("representation"):
                raise ValueError(
                    f"Missing representation metadata for embedding: {options.source_key}"
                )
            self.metadata["embeddings"][name] = {
                "representation": "views",
                "view_name": options.view_name,
                "pooling_name": options.pooling_name,
            }

    def get_metadata(self) -> Dict[str, Any]:
        return deepcopy(self.metadata)

    def extract(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        source_outputs = self.frontend.extract(batch=batch)
        outputs = dict(source_outputs)
        for name, options in self.postprocessing.items():
            if options.source_key not in source_outputs:
                raise KeyError(
                    f"Postprocessing input key not found: {options.source_key}"
                )
            outputs[name] = build_embedding_views(
                embeddings=source_outputs[options.source_key],
                representation=self.source_metadata["embeddings"][options.source_key][
                    "representation"
                ],
                view_name=options.view_name,
                pooling_name=options.pooling_name,
            )
        return outputs
