"""Load the optional rips package and expose type aliases.

The ResInsight interface requires a minimum ``rips`` version that provides the
complete API surface used by xtgeo.  Rather than conditionally importing
individual symbols to support older releases, we enforce an all-or-nothing
version gate: either the installed ``rips`` meets :data:`MIN_RIPS_VERSION` and
every import succeeds, or the user is told to upgrade.
"""

from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any, TypeAlias

from packaging.version import InvalidVersion, Version

if TYPE_CHECKING:
    from types import ModuleType

# Minimum rips version that exposes the full API used by xtgeo
MIN_RIPS_VERSION = "2026.6"


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


def _load_package(package_name: str) -> Any | None:
    """Load a Python package by name, return ``None`` if unavailable."""
    try:
        return importlib.import_module(package_name)
    except ImportError:
        return None


rips = _load_package("rips")
_rips_import_error: str | None = None

if rips is not None:
    try:
        from rips import (
            Case as _RipsCase,
            Instance as _RipsInstance,
            Project as _RipsProject,
            PropertyDataType,  # noqa: F401
            PropertyType,  # noqa: F401
        )
    except ImportError as err:
        _rips_import_error = (
            f"The installed rips package does not provide the required API "
            "symbols (Case, Instance, Project, PropertyDataType, PropertyType): "
            f"{err}. Please upgrade: pip install 'rips>={MIN_RIPS_VERSION}'"
        )
        rips = None
        _RipsCase = Any  # type: ignore[misc,assignment]
        _RipsInstance = Any  # type: ignore[misc,assignment]
        _RipsProject = Any  # type: ignore[misc,assignment]
        PropertyDataType = Any  # type: ignore[misc,assignment]
        PropertyType = Any  # type: ignore[misc,assignment]
else:
    _RipsCase = Any  # type: ignore[misc,assignment]
    _RipsInstance = Any  # type: ignore[misc,assignment]
    _RipsProject = Any  # type: ignore[misc,assignment]
    PropertyDataType = Any  # type: ignore[misc,assignment]
    PropertyType = Any  # type: ignore[misc,assignment]

RipsCaseType: TypeAlias = _RipsCase  # type: ignore[misc]
RipsInstanceType: TypeAlias = _RipsInstance  # type: ignore[misc]
RipsProjectType: TypeAlias = _RipsProject  # type: ignore[misc]

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
