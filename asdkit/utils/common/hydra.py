from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type, TypeVar, cast

from hydra.utils import instantiate as hydra_instantiate
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel

ConfigT = TypeVar("ConfigT", bound=BaseModel)


def instantiate(cfg: Any, *args: Any, recursive: bool = False, **kwargs: Any) -> Any:
    """
    Project-wide default Hydra instantiation.

    By default, Hydra's instantiate() is recursive. This wrapper makes
    non-recursive instantiation the default to reduce accidental nested object
    construction.
    Args:
        cfg: A config node (DictConfig / dataclass / plain dict) with _target_.
        *args: Positional args forwarded to Hydra instantiate.
        recursive: Whether to recursively instantiate nested configs.
        **kwargs: Keyword args forwarded to Hydra instantiate.

    Returns:
        Instantiated object.
    """
    return hydra_instantiate(cfg, *args, _recursive_=recursive, **kwargs)


def hydra_to_pydantic(
    hydra_cfg: DictConfig,
    cls: Type[ConfigT],
    transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
) -> ConfigT:
    """
    Args:
        hydra_cfg: Hydra configuration; interpolations are resolved before validation.
        cls: Pydantic model used for validation.
        transform: Optional preprocessing of the resolved dictionary.

    Returns:
        Validated configuration with the concrete type of cls.
    """
    config_dict = cast(Dict[str, Any], OmegaConf.to_object(hydra_cfg))
    if transform is not None:
        config_dict = transform(config_dict)
    return cls(**config_dict)
