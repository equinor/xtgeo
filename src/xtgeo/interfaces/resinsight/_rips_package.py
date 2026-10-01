"""Load the optional rips package and expose type aliases.

The ResInsight interface requires a minimum ``rips`` version that provides the
complete API surface used by xtgeo. Rather than conditionally importing
individual symbols to support older releases, we enforce an all-or-nothing
version gate: either the installed ``rips`` meets :data:`MIN_RIPS_VERSION` and
every import succeeds, or the user is told to upgrade.

Note the use of TYPE_CHECKING-only imports and runtime Any fallbacks so xtgeo can be
imported without rips, while IDEs and type checkers still get useful hints when rips
is available.
"""

from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias

from packaging.version import InvalidVersion, Version

if TYPE_CHECKING:
    from rips import (
        Case as RipsCaseType,
        Instance as RipsInstanceType,
        NameConflictPolicy as NameConflictPolicyType,
        Project as RipsProjectType,
        PropertyDataType as RipsPropertyDataType,
        PropertyType as RipsPropertyType,
        RegularSurface as RipsRegularSurfaceType,
        RipsError as RipsErrorType,
    )

    class RipsModuleType(Protocol):
        Case: type[RipsCaseType]
        Instance: type[RipsInstanceType]
        Project: type[RipsProjectType]
        PropertyDataType: type[RipsPropertyDataType]
        PropertyType: type[RipsPropertyType]
        NameConflictPolicy: type[NameConflictPolicyType]
        RegularSurface: type[RipsRegularSurfaceType]
        RipsError: type[RipsErrorType]
else:
    NameConflictPolicyType = Any
    RipsPropertyDataType = Any
    RipsPropertyType = Any
    RipsCaseType = Any
    RipsInstanceType = Any
    RipsProjectType = Any
    RipsModuleType = Any
    RipsRegularSurfaceType = Any
    RipsErrorType = Any

# Minimum rips version that exposes the full API used by xtgeo
MIN_RIPS_VERSION = "2026.9"
_REQUIRED_RIPS_SYMBOLS = (
    "Case",
    "Instance",
    "NameConflictPolicy",
    "Project",
    "PropertyDataType",
    "PropertyType",
)


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


def _load_rips_package(package_name: str) -> RipsModuleType | None:
    """Load a Python package by name, return ``None`` if unavailable."""
    try:
        return importlib.import_module(package_name)
    except ImportError:
        return None


rips = _load_rips_package("rips")
_rips_import_error: str | None = None

# Runtime exports of string enums: resolved from rips when available, otherwise Any
NameConflictPolicy: Any = Any
PropertyDataType: Any = Any
PropertyType: Any = Any

if rips is not None:
    missing_symbols = [
        name for name in _REQUIRED_RIPS_SYMBOLS if not hasattr(rips, name)
    ]
    if missing_symbols:
        _rips_import_error = (
            "The installed rips package does not provide the required API "
            f"symbols ({', '.join(missing_symbols)}). "
            f"Please upgrade: pip install 'rips>={MIN_RIPS_VERSION}'"
        )
        rips = None
    else:
        NameConflictPolicy = rips.NameConflictPolicy
        PropertyDataType = rips.PropertyDataType
        PropertyType = rips.PropertyType


ResInsightInstanceOrPortType: TypeAlias = int | RipsInstanceType


def require_rips() -> RipsModuleType:
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
