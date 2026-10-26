from typing import Any, Dict, Optional

import torch

from asdkit.utils.common import instantiate


def build_autoencoder_model(
    model_cfg: Dict[str, Any],
    input_size: Optional[int] = None,
    output_size: Optional[int] = None,
) -> torch.nn.Module:
    """Inject input_size for the encoder or output_size for the decoder."""
    model_kwargs: Dict[str, Any] = {}
    if input_size is not None:
        model_kwargs["input_size"] = input_size
    if output_size is not None:
        model_kwargs["output_size"] = output_size
    return instantiate(cfg=model_cfg, **model_kwargs)
