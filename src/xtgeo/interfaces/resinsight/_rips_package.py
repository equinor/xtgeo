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

RipsCaseType: TypeAlias = Any
RipsInstanceType: TypeAlias = Any
RipsProjectType: TypeAlias = Any

ResInsightInstanceOrPortType: TypeAlias = int | RipsInstanceType
