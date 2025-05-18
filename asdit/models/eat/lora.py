from dataclasses import dataclass
from typing import Literal, Optional, Tuple

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


from ..lora import BaseLoRA, spectrogram_augment
from .EAT.models.mae import interpolate_pos_embed


@dataclass
class UserDirModule:
    user_dir: str


class EATLoRA(BaseLoRA):
    def __init__(
        self,
        ckpt_path: str = "pretrained_models/eat/EAT-base_epoch10_pt.pt",
        lora_cfg: dict = {"r": 64, "target_modules": ["qkv"]},
        embed_size: int = 128,
        last_layer: str = "linear",
        model_cfg: dict = {},
    ):
        super().__init__(
            ckpt_path=ckpt_path,
            lora_cfg=lora_cfg,
            embed_size=embed_size,
            last_layer=last_layer,
            model_cfg=model_cfg,
        )
        self.norm_mean = -4.268
        self.norm_std = 4.569
        if self.last_layer == "attn_stat_pool" and self.prediction_mode == "cls":
            raise ValueError("attn_stat_pool is not supported for cls prediction mode")

    def construct_data2vec(
        self,
        ckpt_path: str,
        sec: float,
        prediction_mode: Literal["cls", "seq"] = "cls",
        specaug: bool = False,
        specaug_freqm: int = 80,
        specaug_timem: int = 80,
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
        self.fbank_params = {
            "htk_compat": True,
            "sample_frequency": 16000,
            "use_energy": False,
            "window_type": "hanning",
            "num_mel_bins": 128,
            "dither": 0.0,
            "frame_shift": 10,
        }
        self.prediction_mode = prediction_mode
        self.specaug = specaug
        if self.specaug:
            self.specaug_freqm = specaug_freqm
            self.specaug_timem = specaug_timem

        self.target_length = ta_kaldi.fbank(
            torch.zeros(1, int(sec * 16000)), **self.fbank_params
        ).shape[0]
        self.target_length = int(np.ceil(self.target_length / 32) * 32)
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
                pretrained_args.model["modalities"]["image"]["start_drop_path_rate"] = (
                    dpr[0]
                )
                pretrained_args.model["modalities"]["image"]["end_drop_path_rate"] = (
                    max(0, dpr[prenet_blocks - 1])
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
                ] = self.target_length
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
            state["model"]["modality_encoders.IMAGE.positional_encoder.positions"] = (
                state["model"]["modality_encoders.IMAGE.positional_encoder.pos_embed"]
            )

            del state["model"]["modality_encoders.IMAGE.positional_encoder.pos_embed"]
        if "modality_encoders.IMAGE.encoder_mask" in state["model"]:
            del state["model"]["modality_encoders.IMAGE.encoder_mask"]
        model.load_state_dict(state["model"], strict=True)
        model.remove_pretraining_modules(modality="image")
        return model

    def construct_model(  # type: ignore
        self,
        ckpt_path: str,
        sec: float,
        prediction_mode: Literal["cls", "seq"] = "cls",
        specaug: bool = False,
        specaug_freqm: int = 80,
        specaug_timem: int = 80,
        **kwargs,
    ) -> Tuple[torch.nn.Module, int]:
        import_user_module(UserDirModule("/".join(__file__.split("/")[:-1] + ["EAT"])))

        model = self.construct_data2vec(
            ckpt_path=ckpt_path,
            sec=sec,
            prediction_mode=prediction_mode,
            specaug=specaug,
            specaug_freqm=specaug_freqm,
            specaug_timem=specaug_timem,
            **kwargs,
        )

        if ckpt_path.split("/")[-1].startswith("EAT-base_"):
            return model, 768
        elif ckpt_path.split("/")[-1].startswith("EAT-large_"):
            return model, 1024
        else:
            raise ValueError(
                f"Unknown checkpoint {ckpt_path}, only EAT-base_* and EAT-large_* are supported"
            )

    def preprocess(self, source: torch.Tensor) -> torch.Tensor:
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
            fbank = ta_kaldi.fbank(waveform, **self.fbank_params)
            fbanks.append(fbank)
        fbank = torch.stack(fbanks, dim=0).unsqueeze(1)  # (B, 1, H, W)

        diff = self.target_length - fbank.shape[-2]
        if diff > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, diff))
            fbank = m(fbank)
        elif diff < 0:
            fbank = fbank[:, 0 : self.target_length, :]
        fbank = (fbank - self.norm_mean) / (self.norm_std * 2)

        return fbank

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args
            x: (B, L)

        Returns
            feats: (B, 1, D) or (B, L, D)
        """
        x = self.preprocess(x)
        if self.training and self.specaug:
            x = spectrogram_augment(
                x,
                specaug_freqm=self.specaug_freqm,
                specaug_timem=self.specaug_timem,
            )
        feats = self.model.extract_features(
            x, mode="IMAGE", mask=False, remove_extra_tokens=False
        )
        if self.prediction_mode == "cls":
            feats = feats["x"][:, :1]  # (B, 1, D)
        elif self.prediction_mode == "seq":
            feats = feats["x"][:, 1:]
        else:
            raise ValueError(
                f"Unknown prediction mode {self.prediction_mode}, only cls and seq are supported"
            )
        return feats
