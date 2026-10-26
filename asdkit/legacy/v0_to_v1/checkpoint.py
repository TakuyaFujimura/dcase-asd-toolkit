"""Convert v0 checkpoints to the v1 frontend-config format."""

import argparse
import logging
import math
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from omegaconf import OmegaConf

from asdkit.legacy.v0_to_v1.config import migrate_saved_config
from asdkit.utils.common.checkpoint import to_checkpoint_config
from asdkit.utils.dcase_utils.parse_sec import parse_sec_cfg

logger = logging.getLogger(__name__)


def migrate_state_dict(
    checkpoint: Dict[str, Any], saved_cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """Translate renamed parameters and add only deterministic legacy buffers."""
    cfg = checkpoint["frontend_cfg"]
    model_cfg = cfg["model_cfg"]
    saved_target = saved_cfg.get("tgt_class", saved_cfg.get("_target_"))
    prefixes: Dict[str, str] = {}
    if saved_target == "asdkit.frontends.AEPLModel":
        prefixes = {
            "network.encoder.": "encoder_model.layers.",
            "network.decoder.": "decoder_model.layers.",
        }
    elif saved_target == "asdkit.frontends.BasicDisPLModel":
        extractor = saved_cfg["model_cfg"].get("extractor_cfg", {})
        extractor_target = extractor.get("tgt_class", extractor.get("_target_"))
        old_backbone = "extractor.model.base_model.model."
        if extractor_target == "asdkit.models.pretrained_models.beats.BEATsLoRA":
            prefixes[old_backbone] = f"{old_backbone}model."
        elif extractor_target == "asdkit.models.pretrained_models.eat.EATLoRA":
            new_backbone = f"{old_backbone}model.model."
            prefixes = {
                f"{old_backbone}modality_encoders.IMAGE.context_encoder.norm.": f"{new_backbone}pre_norm.",
                f"{old_backbone}modality_encoders.IMAGE.": new_backbone,
                old_backbone: new_backbone,
            }
        if prefixes:
            head = model_cfg["embedding_head_cfg"]["_target_"]
            projection = "head." if head.endswith("AttnStatPoolHead") else "projection."
            prefixes["extractor.projection_net."] = f"embedding_head.{projection}"
    state = {}
    for name, value in checkpoint["state_dict"].items():
        for prefix, replacement in prefixes.items():
            if name.startswith(prefix):
                name = f"{replacement}{name[len(prefix):]}"
                break
        state[name] = value
    if saved_target == "asdkit.frontends.AEPLModel":
        from asdkit.models.audio_feature import ConsecutiveStftFrames

        audio_feat = ConsecutiveStftFrames(
            stft_cfg=model_cfg["stft_cfg"], n_frames=model_cfg["n_frames"]
        )
        state.update(audio_feat.state_dict(prefix="audio_feat."))
    loss_cfg = saved_cfg["model_cfg"].get("loss_cfg", {})
    loss_target = loss_cfg.get("tgt_class", loss_cfg.get("_target_"))
    if loss_target == "asdkit.losses.AdaCos":
        for name, value in list(state.items()):
            if name.startswith("head_dict.") and name.endswith(".W"):
                scale_name = f"{name[:-1]}s"
                if scale_name not in state:
                    if loss_cfg.get("dynamic", True):
                        raise ValueError(
                            "Cannot recover an unsaved dynamic AdaCos scale"
                        )
                    state[scale_name] = torch.tensor(
                        math.sqrt(2) * math.log(value.shape[1] - 1), dtype=torch.float64
                    )
    return {**checkpoint, "state_dict": state}


def warn_model_config_conflicts(
    config: Any, parameters: Any, path: str = "model_cfg"
) -> None:
    """Compare shared model settings, allowing keys added during construction."""
    if isinstance(config, Mapping) and isinstance(parameters, Mapping):
        for name, value in parameters.items():
            if name in config:
                warn_model_config_conflicts(
                    config=config[name], parameters=value, path=f"{path}.{name}"
                )
    elif config != parameters:
        logger.warning(
            f"Conflicting frontend setting '{path}': training config={config!r}, "
            f"checkpoint hyper_parameters={parameters!r}; using the checkpoint value."
        )


def upgrade_legacy_checkpoint(
    checkpoint: Dict[str, Any], config_path: Path
) -> Dict[str, Any]:
    """Add frontend metadata in memory; leave the original checkpoint untouched."""
    # Resolve interpolations against the complete training config before selecting frontend.
    training_config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(training_config, dict):
        raise TypeError(
            f"Training config at {config_path} must be a mapping; "
            f"got {type(training_config).__name__}"
        )
    training_config = parse_sec_cfg(training_config)

    # Old load_from_checkpoint reconstructed arguments from saved hyperparameters.
    parameters = checkpoint.get("hyper_parameters", {})

    # Merge before migration: some frontends require a structural config rewrite.
    frontend_cfg = training_config["frontend"]

    # Check for conflicts between the training config and the checkpoint hyperparameters.
    for name, value in parameters.items():
        if name in frontend_cfg:
            if name == "model_cfg":
                warn_model_config_conflicts(config=frontend_cfg[name], parameters=value)
            elif frontend_cfg[name] != value:
                logger.warning(
                    f"Conflicting frontend setting '{name}': "
                    f"training config={frontend_cfg[name]!r}, "
                    f"checkpoint hyper_parameters={value!r}; using the checkpoint value."
                )
        frontend_cfg[name] = value
    upgraded = {**checkpoint, "frontend_cfg": migrate_saved_config(frontend_cfg)}
    upgraded.pop("hyper_parameters", None)
    upgraded.pop("hparams_type", None)
    return migrate_state_dict(checkpoint=upgraded, saved_cfg=frontend_cfg)


def convert_legacy_checkpoint(
    ckpt_path: Union[str, Path],
    output_path: Union[str, Path],
    config_path: Optional[Union[str, Path]] = None,
    weights_only: bool = True,
) -> None:
    """Write a converted copy for later restoration without the training YAML.

    With weights_only=True, convert the full output to supported basic types.
    Otherwise preserve Python types.
    """
    ckpt_path, output_path = Path(ckpt_path), Path(output_path)
    if ckpt_path.resolve() == output_path.resolve():
        raise ValueError("The converted checkpoint must use a different output path")
    if output_path.exists():
        raise FileExistsError(output_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config_path = (
        Path(config_path)
        if config_path is not None
        else ckpt_path.parent.parent / ".hydra" / "config.yaml"
    )
    checkpoint = upgrade_legacy_checkpoint(
        checkpoint=checkpoint, config_path=config_path
    )
    if weights_only:
        checkpoint = to_checkpoint_config(checkpoint)
    buffer = BytesIO()
    torch.save(checkpoint, buffer)
    buffer.seek(0)
    torch.load(buffer, map_location="cpu", weights_only=weights_only)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(buffer.getvalue())


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert a v0 checkpoint to the v1 frontend-config format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ckpt-path", type=Path, required=True, help="Input checkpoint."
    )
    parser.add_argument(
        "--output-path", type=Path, required=True, help="New checkpoint path."
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        help="Training YAML; defaults to <checkpoint parent>/../.hydra/config.yaml.",
    )
    parser.add_argument(
        "--weights-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Convert output to types supported by weights-only loading; disable to preserve Python types.",
    )
    args = parser.parse_args(argv)
    convert_legacy_checkpoint(
        ckpt_path=args.ckpt_path,
        output_path=args.output_path,
        config_path=args.config_path,
        weights_only=args.weights_only,
    )


if __name__ == "__main__":
    main()
