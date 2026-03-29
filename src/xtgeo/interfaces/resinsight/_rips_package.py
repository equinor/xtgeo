"""Load the optional rips package and expose type aliases."""

from __future__ import annotations

import importlib
from typing import Any, TypeAlias


def load_package(package_name: str) -> Any | None:
    """Load a Python package by name, return None if unavailable."""
    try:
        return importlib.import_module(package_name)
    except ImportError:
        return None


rips = load_package("rips")

try:
    from rips import (
        Case as _RipsCase,
        Instance as _RipsInstance,
        Project as _RipsProject,
    )
except ImportError:
    _RipsCase = Any  # type: ignore[misc,assignment]
    _RipsInstance = Any  # type: ignore[misc,assignment]
    _RipsProject = Any  # type: ignore[misc,assignment]

RipsCaseType: TypeAlias = _RipsCase  # type: ignore[misc]
RipsInstanceType: TypeAlias = _RipsInstance  # type: ignore[misc]
RipsProjectType: TypeAlias = _RipsProject  # type: ignore[misc]

ResInsightInstanceOrPortType: TypeAlias = int | RipsInstanceType
