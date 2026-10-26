from pathlib import Path
from typing import Any, Dict, Optional, final

import torch

from asdkit.models.audio_feature import ConsecutiveStftFrames
from asdkit.utils.common import instantiate

from ..base_generative import BaseGenerativePLFrontend
from .utils import build_autoencoder_model


class Ae1DStftFramesPLFrontend(BaseGenerativePLFrontend):
    def __init__(
        self,
        model_cfg: Dict[str, Any],
        optim_cfg: Dict[str, Any],
        lrscheduler_cfg: Optional[Dict[str, Any]] = None,
        label_dict_path: Optional[Dict[str, Path]] = None,
        save_only_trainable: bool = False,
        optimizer_log_options: Optional[Dict[str, bool]] = None,
        use_precomputed_features_for_training: bool = False,
    ) -> None:
        super().__init__(
            model_cfg=model_cfg,
            optim_cfg=optim_cfg,
            lrscheduler_cfg=lrscheduler_cfg,
            label_dict_path=label_dict_path,
            save_only_trainable=save_only_trainable,
            optimizer_log_options=optimizer_log_options,
        )
        self.use_precomputed_features_for_training = (
            use_precomputed_features_for_training
        )
        self.construct_model(**model_cfg)
        if self.save_only_trainable:
            self.save_state_names = self.collect_trainable_state_names()

    def get_metadata(self) -> Dict[str, Any]:
        """
        Returns:
            Metadata marking ``embed`` as a vector.
        """
        return {"embeddings": {"embed": {"representation": "vector"}}}

    @final
    def construct_model(
        self,
        encoder_model_cfg: Dict[str, Any],
        decoder_model_cfg: Dict[str, Any],
        stft_cfg: Dict[str, Any],
        n_frames: int,
        loss_cfg: Dict[str, Any],
        frame_shift: Optional[int] = None,
    ) -> None:
        """
        Args:
            encoder_model_cfg: Encoder configuration.
            decoder_model_cfg: Decoder configuration.
            stft_cfg: Spectrogram extractor configuration.
            n_frames: Consecutive frames concatenated per feature vector.
            loss_cfg: Reconstruction loss configuration.
            frame_shift: Explicit unpadded window stride; None keeps tail padding.
        """
        self.audio_feat = ConsecutiveStftFrames(
            stft_cfg=stft_cfg, n_frames=n_frames, frame_shift=frame_shift
        )
        self.encoder_model = build_autoencoder_model(
            model_cfg=encoder_model_cfg,
            input_size=self.audio_feat.dim,
        )
        self.decoder_model = build_autoencoder_model(
            model_cfg=decoder_model_cfg,
            output_size=self.audio_feat.dim,
        )
        self.loss = instantiate(cfg=loss_cfg, reduction="none")

    def compute_outputs(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: wave shaped (B, samples).
                With use_precomputed_features_for_training enabled, training requires
                feat shaped (B, D). Each feature row is one training sample.

        B counts waveforms or precomputed windows; n_vectors counts windows
        per sample, and feat_dim is n_frames times the frequency-bin count.

        Returns:
            loss_main and AS-main have shape (B,); embed has shape
            (B, embedding_dim), averaged over time.
        """
        # ConsecutiveStftFrames groups consecutive frames into windows,
        # returning [B, n_vectors, feat_dim].
        if self.training and self.use_precomputed_features_for_training:
            assert batch["feat"].ndim == 2
            assert batch["feat"].shape[-1] == self.audio_feat.dim
            # ConsecutiveStftFramesDataset precomputes individual windows.
            # The DataLoader shuffles them across audio files into [B, feat_dim].
            x_ref = batch["feat"].unsqueeze(1)  # [B, 1, feat_dim]
        else:
            x_ref = self.audio_feat(batch["wave"])  # [B, n_vectors, feat_dim]
        B, n_vectors, _ = x_ref.shape
        x_ref = x_ref.reshape(B * n_vectors, -1)  # [B*n_vectors, feat_dim]
        z = self.encoder_model(x_ref)  # [B*n_vectors, z_dim]
        x_est = self.decoder_model(z)
        embed = z.reshape(B, n_vectors, z.shape[-1]).mean(dim=1)  # [B, z_dim]

        loss_value = self.loss(x_est, x_ref).view(B, -1).mean(dim=-1)

        return {"loss_main": loss_value, "AS-main": loss_value, "embed": embed}
