"""Translate configuration names from the pre-refactor v0 format."""

from pathlib import Path
from typing import Any, Dict

from omegaconf import OmegaConf


def migrate_lora_model(model: Dict[str, Any]) -> Dict[str, Any]:
    """Split the saved projected extractor into a standard backbone and head."""
    extractor = model["extractor_cfg"]
    target = extractor.get("_target_")
    if target not in {
        "asdkit.models.pretrained_models.beats.BEATsLoRA",
        "asdkit.models.pretrained_models.eat.EATLoRA",
    }:
        return model
    backbone = "beats" if ".beats." in target else "eat"
    saved = extractor["model_cfg"]
    if saved.get("specaug", False):
        raise ValueError(f"Cannot migrate {target} with specaug enabled")
    backbone_cfg: Dict[str, Any] = {
        "_target_": (
            f"asdkit.models.pretrained_models.{backbone}."
            f"{'BEATsSSLModel' if backbone == 'beats' else 'EATSSLModel'}"
        ),
        "sr": saved.get("sr", 16000),
        "spectrogram_augmentation_cfg": None,
    }
    if backbone == "beats":
        path = Path(saved["ckpt_path"])
        if path.parent.name != "beats":
            raise ValueError(
                f"BEATs ckpt_path must be under a 'beats' directory; got {str(path)!r}"
            )
        backbone_cfg.update(
            model_id=path.stem,
            cache_dir=str(path.parent.parent),
            update_cfg=saved.get("update_cfg"),
        )
    else:
        if "EAT-base" not in saved["ckpt_path"]:
            raise ValueError(
                f"Only EAT-base checkpoints can be migrated; got {saved['ckpt_path']!r}"
            )
        if saved.get("update_cfg"):
            raise ValueError(
                f"Cannot migrate EAT with update_cfg overrides: {saved['update_cfg']!r}"
            )
        if saved.get("prediction_mode", "cls") not in {"cls", "seq"}:
            raise ValueError(
                "EAT prediction_mode must be 'cls' or 'seq'; "
                f"got {saved['prediction_mode']!r}"
            )
        backbone_cfg["model_id"] = "worstchan/EAT-base_epoch30_pretrain"
    projection = extractor["projection_type"]
    if projection not in {"linear", "attn_stat_pool"}:
        raise ValueError(
            "LoRA projection_type must be 'linear' or 'attn_stat_pool'; "
            f"got {projection!r}"
        )
    mode = "seq"
    if backbone == "eat" and saved.get("prediction_mode", "cls") == "cls":
        mode = "cls"
    head = "AttnStatPoolHead" if projection == "attn_stat_pool" else "MeanPoolingHead"
    if mode == "cls":
        if projection != "linear":
            raise ValueError(
                "EAT prediction_mode='cls' requires projection_type='linear'; "
                f"got {projection!r}"
            )
        head = "LinearHead"
    return {
        **model,
        "extractor_cfg": {
            "_target_": "asdkit.models.pretrained_models.wrapper.PretrainedWrapper",
            "model_cfg": backbone_cfg,
            "adaptation_cfg": {
                "_target_": "asdkit.models.pretrained_models.adaptation.LoRA",
                "lora_cfg": extractor["lora_cfg"],
            },
        },
        "embedding_head_cfg": {
            "_target_": f"asdkit.models.pretrained_models.heads.{head}",
            "embedding_dim": extractor["embed_size"],
        },
        "head_input_mode": mode,
    }


def migrate_saved_config(config: Any) -> Any:
    """
    Args:
        config: Saved configuration or nested value using current or pre-refactor names.
            OmegaConf containers are resolved without converting value types.

    Returns:
        Configuration with current names, or an unchanged leaf value.
        The input is unchanged. This conversion is limited to checkpoint restore.
    """
    if OmegaConf.is_config(config):
        config = OmegaConf.to_container(config, resolve=True)
    if isinstance(config, list):
        return [migrate_saved_config(item) for item in config]
    if not isinstance(config, dict):
        return config
    saved_target = config.get("tgt_class", config.get("_target_"))
    migrated: Dict[str, Any] = {}
    for key, value in config.items():
        if key == "use_compile":
            continue
        migrated["_target_" if key == "tgt_class" else key] = migrate_saved_config(
            value
        )

    if saved_target in {
        "asdkit.losses.SCAdaCos",
        "asdkit.losses.scadacos.SCAdaCos",
    }:
        migrated["_target_"] = "asdkit.losses.SubClusterAdaCos"
    elif saved_target == "asdkit.frontends.AEPLModel":
        model = migrated["model_cfg"]
        network = model["network_cfg"]
        if network["_target_"] != "asdkit.models.autoencoder.AE_MLP":
            raise ValueError(
                "AEPLModel migration requires network_cfg._target_="
                f"'asdkit.models.autoencoder.AE_MLP'; got {network['_target_']!r}"
            )
        audio = model["audio_feat_cfg"]
        if audio["_target_"] != "asdkit.models.audio_feature.FlattenSTFT":
            raise ValueError(
                "AEPLModel migration requires audio_feat_cfg._target_="
                f"'asdkit.models.audio_feature.FlattenSTFT'; got {audio['_target_']!r}"
            )
        shared = {key: value for key, value in network.items() if key != "_target_"}
        migrated["_target_"] = "asdkit.frontends.Ae1DStftFramesPLFrontend"
        migrated["use_precomputed_features_for_training"] = True
        migrated["model_cfg"] = {
            "encoder_model_cfg": {
                "_target_": "asdkit.models.autoencoder.DCASEMLPEncoder",
                **shared,
            },
            "decoder_model_cfg": {
                "_target_": "asdkit.models.autoencoder.DCASEMLPDecoder",
                **shared,
            },
            "stft_cfg": audio["stft_cfg"],
            "n_frames": audio["n_frames"],
            "frame_shift": 1,
            "loss_cfg": model["loss_cfg"],
        }
    elif saved_target in {
        "asdkit.frontends.BasicDisPLModel",
        "asdkit.frontends.FeatExPLModel",
        "asdkit.frontends.SubspaceLossPLModel",
    }:
        model = migrated["model_cfg"]
        if "label_to_lossweight_dict" in model:
            model["label_loss_weight_dict"] = model.pop("label_to_lossweight_dict")
        if saved_target == "asdkit.frontends.BasicDisPLModel":
            migrated["_target_"] = "asdkit.frontends.BasicDisPLFrontend"
            extractor_target = model.get("extractor_cfg", {}).get("_target_")
            if extractor_target in {
                "asdkit.models.pretrained_models.beats.BEATsLoRA",
                "asdkit.models.pretrained_models.eat.EATLoRA",
            }:
                migrated["model_cfg"] = migrate_lora_model(model)
                migrated["_target_"] = "asdkit.frontends.PretrainedDisPLFrontend"
        else:
            if "subspace_embed_size" in model:
                model["subspace_embedding_dim"] = model.pop("subspace_embed_size")
            migrated["_target_"] = (
                "asdkit.frontends.FeatExPLFrontend"
                if saved_target == "asdkit.frontends.FeatExPLModel"
                else "asdkit.frontends.SubspaceLossPLFrontend"
            )
    elif saved_target in {
        "asdkit.augmentations.FeatEx",
        "asdkit.augmentations.featex.FeatEx",
    }:
        if "subspace_embed_size" in migrated:
            migrated["subspace_embedding_dim"] = migrated.pop("subspace_embed_size")
    elif (
        saved_target == "asdkit.models.pretrained_models.wrapper.BasePretrainedWrapper"
    ):
        migrated["_target_"] = (
            "asdkit.models.pretrained_models.wrapper.PretrainedWrapper"
        )
    else:
        frontend_names = {
            "asdkit.frontends.Ae1DStftFramesPLModel": "asdkit.frontends.Ae1DStftFramesPLFrontend",
        }
        if saved_target in frontend_names:
            migrated["_target_"] = frontend_names[saved_target]
    return migrated
