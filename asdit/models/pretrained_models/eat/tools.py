from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torchaudio.compliance.kaldi as ta_kaldi
from omegaconf import open_dict

try:
    from fairseq import checkpoint_utils, tasks
    from fairseq.utils import import_user_module
except ImportError:
    print("----------------------------------------------")
    print("[Error] Please install fairseq to use EATLoRA.")
    print("----------------------------------------------")


from .EAT.models.mae import interpolate_pos_embed

FBANK_PRAMS = {
    "htk_compat": True,
    "sample_frequency": 16000,
    "use_energy": False,
    "window_type": "hanning",
    "num_mel_bins": 128,
    "dither": 0.0,
    "frame_shift": 10,
}


def calc_target_length(sec: float, sr: int) -> int:
    dummy_fbank = ta_kaldi.fbank(torch.zeros(1, int(sec * sr)), **FBANK_PRAMS)
    target_length = int(np.ceil(dummy_fbank.shape[0] / 32) * 32)
    return target_length


def preprocess(
    source: torch.Tensor,
    target_length: int,
    norm_mean: float = -4.268,
    norm_std: float = 4.569,
) -> torch.Tensor:
    """
    Args:
        source (torch.Tensor): (B, L)

    Returns:
        fbank (torch.Tensor): (B, H, W)
    """
    fbanks = []
    source = source - source.mean(dim=-1, keepdim=True)  # (B, L)
    for waveform in source:
        waveform = waveform.unsqueeze(0)  # (1, L)
        fbank = ta_kaldi.fbank(waveform, **FBANK_PRAMS)
        fbanks.append(fbank)
    fbank = torch.stack(fbanks, dim=0).unsqueeze(1)  # (B, 1, H, W)

    diff = target_length - fbank.shape[-2]
    if diff > 0:
        m = torch.nn.ZeroPad2d((0, 0, 0, diff))
        fbank = m(fbank)
    elif diff < 0:
        raise ValueError("The input length is longer than the target_length")
        # fbank = fbank[:, 0 : self.target_length, :]
    fbank = (fbank - norm_mean) / (norm_std * 2)

    return fbank


@dataclass
class UserDirModule:
    user_dir: str


def construct_data2vec(
    ckpt_path: Path | str,
    sec: float,
    sr: int,
    drop_path_rate: float = 0.1,
    norm_eps: Optional[float] = None,
    remove_alibi: bool = False,
    encoder_dropout: float = 0.0,
    post_mlp_drop: float = 0.0,
    attention_dropout: float = 0.0,
    activation_dropout: float = 0.0,
    dropout_input: float = 0.0,
    layerdrop: float = 0.0,
    prenet_layerdrop: float = 0.0,
    prenet_dropout: float = 0.0,
):
    if sr != 16000:
        raise ValueError("The sampling rate should be 16000")

    target_length = calc_target_length(sec=sec, sr=sr)

    # adjust pre-training config into fine-tuning
    state = checkpoint_utils.load_checkpoint_to_cpu(ckpt_path, {})
    pretrained_args = state.get("cfg", None)
    pretrained_args.criterion = None
    pretrained_args.lr_scheduler = None
    with open_dict(pretrained_args.model):
        pretrained_args.model.drop_path_rate = drop_path_rate
        if norm_eps is not None:
            pretrained_args.model.norm_eps = norm_eps

    if "modalities" in pretrained_args.model:
        prenet_blocks = pretrained_args.model["modalities"]["image"]["prenet_depth"]
        model_blocks = pretrained_args.model["depth"]
        with open_dict(pretrained_args):
            dpr = np.linspace(0, drop_path_rate, model_blocks).tolist()
            pretrained_args.model["modalities"]["image"]["start_drop_path_rate"] = dpr[
                0
            ]
            pretrained_args.model["modalities"]["image"]["end_drop_path_rate"] = max(
                0, dpr[prenet_blocks - 1]
            )
            pretrained_args.model["start_drop_path_rate"] = dpr[prenet_blocks]
            pretrained_args.model["end_drop_path_rate"] = dpr[-1]

            if "mae_masking" in pretrained_args.model["modalities"]["image"]:
                del pretrained_args.model["modalities"]["image"]["mae_masking"]

            if remove_alibi:
                pretrained_args.model["modalities"]["image"][
                    "use_alibi_encoder"
                ] = False
                if (
                    state is not None
                    and "modality_encoders.IMAGE.alibi_bias" in state["model"]
                ):
                    del state["model"]["modality_encoders.IMAGE.alibi_bias"]

            pretrained_args.model["encoder_dropout"] = encoder_dropout
            pretrained_args.model["post_mlp_drop"] = post_mlp_drop
            pretrained_args.model["attention_dropout"] = attention_dropout
            pretrained_args.model["activation_dropout"] = activation_dropout
            pretrained_args.model["dropout_input"] = dropout_input
            pretrained_args.model["layerdrop"] = layerdrop

            pretrained_args.model["modalities"]["image"][
                "prenet_layerdrop"
            ] = prenet_layerdrop
            pretrained_args.model["modalities"]["image"][
                "prenet_dropout"
            ] = prenet_dropout

            pretrained_args.model["modalities"]["image"][
                "target_length"
            ] = target_length
    else:
        # not d2v multi
        with open_dict(pretrained_args):
            pretrained_args.model["drop_path_rate"] = drop_path_rate
            pretrained_args.model["block_dropout"] = encoder_dropout
            pretrained_args.model["attention_dropout"] = attention_dropout
            pretrained_args.model["activation_dropout"] = activation_dropout

    task = tasks.setup_task(pretrained_args.task)
    model = task.build_model(pretrained_args.model, from_checkpoint=True)

    assert "data2vec_multi" in pretrained_args.model._name

    # adjust position embedding for specific downstream task (due to different fixed clip length)
    interpolate_pos_embed(model, state)
    if "modality_encoders.IMAGE.positional_encoder.pos_embed" in state["model"]:
        state["model"]["modality_encoders.IMAGE.positional_encoder.positions"] = state[
            "model"
        ]["modality_encoders.IMAGE.positional_encoder.pos_embed"]

        del state["model"]["modality_encoders.IMAGE.positional_encoder.pos_embed"]
    if "modality_encoders.IMAGE.encoder_mask" in state["model"]:
        del state["model"]["modality_encoders.IMAGE.encoder_mask"]
    model.load_state_dict(state["model"], strict=True)
    model.remove_pretraining_modules(modality="image")
    return model


def restore(
    ckpt_path: Path | str, update_cfg: Optional[dict] = None
) -> Tuple[torch.nn.Module, int]:
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. Please download it."
        )
    if update_cfg is None:
        update_cfg = {}
    import_user_module(UserDirModule("/".join(__file__.split("/")[:-1] + ["EAT"])))
    model = construct_data2vec(ckpt_path=ckpt_path, **update_cfg)
    ckpt_path = str(ckpt_path)
    if ckpt_path.split("/")[-1].startswith("EAT-base_"):
        return model, 768
    elif ckpt_path.split("/")[-1].startswith("EAT-large_"):
        return model, 1024
    else:
        raise ValueError(
            f"Unknown checkpoint {ckpt_path}, only EAT-base_* and EAT-large_* are supported"
        )
