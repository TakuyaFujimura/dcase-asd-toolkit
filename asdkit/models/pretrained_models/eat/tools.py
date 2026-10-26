import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torchaudio.compliance.kaldi as ta_kaldi

from asdkit.utils.common.optional_dependencies import require_optional

logger = logging.getLogger(__name__)

FBANK_PARAMS = {
    "htk_compat": True,
    "sample_frequency": 16000,
    "use_energy": False,
    "window_type": "hanning",
    "num_mel_bins": 128,
    "dither": 0.0,
    "frame_shift": 10,
}


def preprocess(
    source: torch.Tensor,
    norm_mean: float = -4.268,
    norm_std: float = 4.569,
    pad_to_multiple: Optional[int] = 32,
) -> torch.Tensor:
    """
    Args:
        source (torch.Tensor): Waveforms with shape (B, L).
        pad_to_multiple: Zero-pad time frames to this multiple before normalization.
            None disables padding.

    Returns:
        fbank (torch.Tensor): (B, 1, T, F).
    """
    assert source.ndim == 2
    if pad_to_multiple is not None and (
        not isinstance(pad_to_multiple, int)
        or isinstance(pad_to_multiple, bool)
        or pad_to_multiple <= 0
    ):
        raise ValueError("pad_to_multiple must be a positive integer or None")
    fbanks = []
    source = source - source.mean(dim=-1, keepdim=True)  # (B, L)
    for waveform in source:
        waveform = waveform.unsqueeze(0)  # (1, L)
        fbank = ta_kaldi.fbank(waveform, **FBANK_PARAMS)
        fbanks.append(fbank)
    fbank = torch.stack(fbanks, dim=0).unsqueeze(1)  # (B, 1, T, F)

    if pad_to_multiple is not None:
        padding = (-fbank.shape[-2]) % pad_to_multiple
        fbank = torch.nn.functional.pad(fbank, (0, 0, 0, padding))

    fbank = (fbank - norm_mean) / (norm_std * 2)

    return fbank


def restore(
    model_id: Union[Path, str], cache_dir: Optional[Union[str, Path]] = None
) -> Tuple[torch.nn.Module, int]:
    transformers = require_optional(
        "transformers", "transformers", "eat", "EAT", ">=4.23,<5"
    )
    model = transformers.AutoModel.from_pretrained(
        model_id, cache_dir=cache_dir, trust_remote_code=True
    )

    embedding_dim = model.model.config.embed_dim
    logger.info(f"Restored EAT model from {model_id}, embedding_dim: {embedding_dim}")
    return model, embedding_dim
