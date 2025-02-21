import asdlib
from hydra.utils import instantiate


def instantiate_tgt(config_org: dict):
    if "_target_" in config_org:
        raise ValueError(
            "'_target_' key should be replaced with 'tgt_class' key. "
            + "(Pydantic does not allow keys starting with '_'.)"
        )
    if "tgt_class" not in config_org:
        raise ValueError("tgt_class key is missing")
    config = config_org.copy()
    tgt_class = config.pop("tgt_class")
    model = instantiate({"_target_": tgt_class, **config})
    return model
