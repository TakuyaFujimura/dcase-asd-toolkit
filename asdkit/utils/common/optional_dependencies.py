"""Load optional dependencies, checking versions only when requested."""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from types import ModuleType

from packaging.specifiers import SpecifierSet


def require_optional(
    module: str,
    distribution: str,
    extra: str,
    feature: str,
    specifier: str = "",
) -> ModuleType:
    """Import a dependency, optionally check its version, and suggest an extra."""
    requirement = f"{distribution}{specifier}"
    install_hint = f"Run `uv sync --extra {extra}` or `pip install 'asdkit[{extra}]'`."
    if specifier:
        try:
            installed = version(distribution)
        except PackageNotFoundError as exc:
            raise ImportError(
                f"{feature} requires {requirement}. {install_hint}"
            ) from exc
        if installed not in SpecifierSet(specifier):
            raise ImportError(
                f"{feature} requires {requirement}, but {distribution} {installed} "
                f"is installed. {install_hint}"
            )
    try:
        return import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name != module.split(".")[0]:
            raise
        raise ImportError(f"{feature} requires {requirement}. {install_hint}") from exc
