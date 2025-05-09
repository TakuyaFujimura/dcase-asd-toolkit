from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as ta_kaldi
from omegaconf import open_dict

from fairseq import checkpoint_utils, tasks
from fairseq.utils import import_user_module

from ..lora import BaseLoRA
from .EAT.models.mae import interpolate_pos_embed

# from fairseq.checkpoint_utils import load_model_ensemble_and_task


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

    def construct_model(
        self,
        ckpt_path: str,
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
        target_length: int = 1024,
        specaug: bool = False,
        specaug_freqm: int = 80,
        specaug_timem: int = 80,
    ):  # type: ignore
        import_user_module(UserDirModule("/".join(__file__.split("/")[:-1] + ["EAT"])))

        self.specaug = specaug
        if self.specaug:
            self.specaug_freqm = specaug_freqm
            self.specaug_timem = specaug_timem

        self.target_length = target_length

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
            state["model"]["modality_encoders.IMAGE.positional_encoder.positions"] = (
                state["model"]["modality_encoders.IMAGE.positional_encoder.pos_embed"]
            )

            del state["model"]["modality_encoders.IMAGE.positional_encoder.pos_embed"]
        if "modality_encoders.IMAGE.encoder_mask" in state["model"]:
            del state["model"]["modality_encoders.IMAGE.encoder_mask"]
        model.load_state_dict(state["model"], strict=True)
        model.remove_pretraining_modules(modality="image")

        return model, 768

        # model, cfg, task = load_model_ensemble_and_task([ckpt_path])
        # return model[0], 768

        # specaug

    def spectrogram_augment(self, spec):
        freq_masking = torchaudio.transforms.FrequencyMasking(
            self.specaug_freqm, iid_masks=True
        )
        time_masking = torchaudio.transforms.TimeMasking(
            self.specaug_timem, iid_masks=True
        )
        spec_ = spec.transpose(2, 3)
        input_with_freq_mask = freq_masking(spec_)
        input_with_time_freq_mask = time_masking(input_with_freq_mask)
        input_with_time_freq_mask = torch.transpose(input_with_time_freq_mask, 2, 3)
        return input_with_time_freq_mask

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
            fbank = ta_kaldi.fbank(
                waveform,
                htk_compat=True,
                sample_frequency=16000,
                use_energy=False,
                window_type="hanning",
                num_mel_bins=128,
                dither=0.0,
                frame_shift=10,
            )
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

    def extract_features(self, x):
        """
        Args
            x: (B, L)
        """
        x = self.preprocess(x)
        if self.training and self.specaug:
            x = self.spectrogram_augment(x)
        feats = self.model.extract_features(
            x, mode="IMAGE", mask=False, remove_extra_tokens=False
        )
        feats = feats["x"][:, :1]  # (B, 1, D)
        return feats

    # def _load_from_state_dict(
    #     self,
    #     state_dict: dict,
    #     prefix: str,
    #     local_metadata: dict,
    #     strict: bool,
    #     missing_keys: list,
    #     unexpected_keys: list,
    #     error_msgs: list,
    # ):
    #     # new_state_dict = state_dict
    #     super()._load_from_state_dict(
    #         state_dict,
    #         prefix="",
    #         local_metadata=local_metadata,
    #         strict=strict,
    #         missing_keys=missing_keys,
    #         unexpected_keys=unexpected_keys,
    #         error_msgs=error_msgs,
    #     )
