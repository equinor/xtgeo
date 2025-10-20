"""Loading rmsapi package based on availability and add specific RMS type annotations.

As xtgeo from version 4 only support RMSAPI > 1.10, we do not support roxar
package anymore. This module handles the optional import of the rmsapi package
and its submodules, as well as type annotations for rmsapi classes.

"""

from __future__ import annotations

import importlib
import pathlib
from typing import Any, Optional, TypeAlias, Union


def load_package(package_name: str) -> Optional[Any]:
    """Load a Python package by name, return None if not available."""
    try:
        return importlib.import_module(package_name)
    except ImportError:
        return None


# Load main packages
rmsapi = load_package("rmsapi")


# Load submodules from the available package
rmsapi_well_picks = load_package("rmsapi.well_picks")
rmsapi_jobs = load_package("rmsapi.jobs")
rmsapi_grids = load_package("rmsapi.grids")

try:
    from rmsapi import (
        Project as _RmsProject,
        Surface as _RmsSurface,
        VerticalDomain as _RmsVerticalDomain,
    )
    from rmsapi.grids import Grid3D as _RmsGrid3D
    from rmsapi.jobs import Jobs as _RmsJobs
    from rmsapi.well_picks import (
        WellPick as _RmsWellPick,
        WellPickAttribute as _RmsWellPickAttribute,
        WellPicks as _RmsWellPicks,
        WellPickSet as _RmsWellPickSet,
    )
except ImportError:
    # Fallback when rmsapi is not available
    _RmsProject = Any  # type: ignore[misc]
    _RmsSurface = Any  # type: ignore[misc]
    _RmsVerticalDomain = Any  # type: ignore[misc]
    _RmsGrid3D = Any  # type: ignore[misc]
    _RmsJobs = Any  # type: ignore[misc]
    _RmsWellPicks = Any  # type: ignore[misc]
    _RmsWellPick = Any  # type: ignore[misc]
    _RmsWellPickAttribute = Any  # type: ignore[misc]
    _RmsWellPickSet = Any  # type: ignore[misc]

RmsProjectType: TypeAlias = _RmsProject  # type: ignore[misc]
RmsSurfaceType: TypeAlias = _RmsSurface  # type: ignore[misc]
RmsVerticalDomainType: TypeAlias = _RmsVerticalDomain  # type: ignore[misc]
RmsGrid3DType: TypeAlias = _RmsGrid3D  # type: ignore[misc]
RmsJobsType: TypeAlias = _RmsJobs  # type: ignore[misc]
RmsWellPicksType: TypeAlias = _RmsWellPicks  # type: ignore[misc]
RmsWellPickType: TypeAlias = _RmsWellPick  # type: ignore[misc]
RmsWellPickAttributeType: TypeAlias = _RmsWellPickAttribute  # type: ignore[misc]
RmsWellPickSetType: TypeAlias = _RmsWellPickSet  # type: ignore[misc]

RmsProjectOrPathType: TypeAlias = Union[str, pathlib.Path, RmsProjectType]
