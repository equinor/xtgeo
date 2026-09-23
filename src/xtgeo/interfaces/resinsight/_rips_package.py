"""Load the optional rips package and expose type aliases.

The ResInsight interface requires a minimum ``rips`` version that provides the
complete API surface used by xtgeo.  Rather than conditionally importing
individual symbols to support older releases, we enforce an all-or-nothing
version gate: either the installed ``rips`` meets :data:`MIN_RIPS_VERSION` and
every required API symbol is available, or the user is told to upgrade.
"""

from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any, TypeAlias

from packaging.version import InvalidVersion, Version

if TYPE_CHECKING:
    from types import ModuleType

    import rips as rips  # type: ignore[import-not-found]
    from rips import (  # type: ignore[import-not-found]
        Case as _RipsCase,
        Instance as _RipsInstance,
        NameConflictPolicy,
        Project as _RipsProject,
        PropertyDataType,
        PropertyType,
        RegularSurface as _RipsRegularSurface,
        SurfaceCollection as _RipsSurfaceCollection,
    )

# Minimum rips version that exposes the full API used by xtgeo
MIN_RIPS_VERSION = "2026.9"


def _check_rips_version() -> None:
    """Raise ``RuntimeError`` if the installed rips version is too old or unparseable.

    Called at runtime (from :func:`require_rips`) rather than at import time so
    that the module can be loaded and inspected even when the installed ``rips``
    version does not meet the minimum requirement.
    """
    try:
        installed = version("rips")
    except PackageNotFoundError:
        raise RuntimeError(
            "rips package is installed but its version metadata is missing. "
            f"Please reinstall: pip install 'rips>={MIN_RIPS_VERSION}'"
        ) from None

    try:
        installed_ver = Version(installed)
    except InvalidVersion:
        raise RuntimeError(
            f"rips package reports version '{installed}' which is not a valid "
            f"PEP 440 version string. Please install a supported release: "
            f"pip install 'rips>={MIN_RIPS_VERSION}'"
        ) from None

    if installed_ver < Version(MIN_RIPS_VERSION):
        raise RuntimeError(
            f"xtgeo requires rips >= {MIN_RIPS_VERSION}, "
            f"but {installed} is installed. "
            f"Please upgrade: pip install 'rips>={MIN_RIPS_VERSION}'"
        )


def _load_package(package_name: str) -> ModuleType | None:
    """Load a Python package by name.

    Return ``None`` if the requested package is unavailable. Import errors raised
    from inside an installed package, including missing transitive dependencies,
    propagate unchanged so users see the underlying installation problem instead
    of a misleading message that the requested package is unavailable.
    """
    try:
        return importlib.import_module(package_name)
    except ModuleNotFoundError as err:
        if err.name == package_name:
            return None
        raise


_RIPS_API_SYMBOLS = (
    "Case",
    "Instance",
    "NameConflictPolicy",
    "Project",
    "PropertyDataType",
    "PropertyType",
    "RegularSurface",
    "RipsError",
    "SurfaceCollection",
)


def _load_rips_api(package: ModuleType) -> dict[str, Any]:
    """Return the required API symbols or raise ``ImportError``."""
    try:
        return {name: getattr(package, name) for name in _RIPS_API_SYMBOLS}
    except AttributeError as err:
        raise ImportError(str(err)) from err


def _initialize_rips_api(
    package: ModuleType | None,
) -> tuple[ModuleType | None, str | None, dict[str, Any]]:
    """Return state for an absent, complete, or incomplete rips package."""
    if package is None:
        return None, None, {}
    try:
        api = _load_rips_api(package)
    except ImportError as err:
        required_symbols = ", ".join(_RIPS_API_SYMBOLS)
        error = (
            f"The installed rips package does not provide the required API "
            f"symbols ({required_symbols}): {err}. "
            f"Please upgrade: pip install 'rips>={MIN_RIPS_VERSION}'"
        )
        return None, error, {}
    return package, None, api


def _rips_symbol(api: dict[str, Any], name: str) -> Any:
    """Return a loaded rips symbol, falling back to ``Any``."""
    return api.get(name, Any)


_rips_import_error: str | None

# Stable annotation names for optional rips:
# When rips exists, they refer to its real classes.
# When rips is unavailable at runtime, they fall back to Any,
# allowing the rest of XTGeo to import.
if not TYPE_CHECKING:  # pragma: no branch
    rips, _rips_import_error, _rips_api = _initialize_rips_api(_load_package("rips"))
    _RipsCase = _rips_symbol(_rips_api, "Case")
    _RipsInstance = _rips_symbol(_rips_api, "Instance")
    NameConflictPolicy = _rips_symbol(_rips_api, "NameConflictPolicy")
    _RipsProject = _rips_symbol(_rips_api, "Project")
    PropertyDataType = _rips_symbol(_rips_api, "PropertyDataType")
    PropertyType = _rips_symbol(_rips_api, "PropertyType")
    _RipsRegularSurface = _rips_symbol(_rips_api, "RegularSurface")
    _RipsSurfaceCollection = _rips_symbol(_rips_api, "SurfaceCollection")

RipsCaseType: TypeAlias = _RipsCase  # type: ignore[misc]
RipsInstanceType: TypeAlias = _RipsInstance  # type: ignore[misc]
RipsProjectType: TypeAlias = _RipsProject  # type: ignore[misc]
RipsRegularSurfaceType: TypeAlias = _RipsRegularSurface  # type: ignore[misc]
RipsSurfaceCollectionType: TypeAlias = _RipsSurfaceCollection  # type: ignore[misc]

ResInsightInstanceOrPortType: TypeAlias = int | RipsInstanceType


def require_rips() -> ModuleType:
    """Return the ``rips`` module or raise ``RuntimeError`` if unavailable/too old.

    Call this at the top of any function that requires the rips package.
    The return value is the validated ``rips`` module, which eliminates
    the need for ``assert rips is not None`` type-narrowing at each call
    site.
    """
    if rips is None:
        raise RuntimeError(
            _rips_import_error
            or (
                "rips package is not available. Please install "
                f"rips >= {MIN_RIPS_VERSION} to use ResInsight features."
            )
        )
    _check_rips_version()
    return rips
