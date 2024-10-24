"""Loading rmsapi or roxar module based on availability"""

import importlib
from typing import TYPE_CHECKING, Any, Optional, TypeVar

# Roxar*Type are placeholders for the actual types which are not known at this point,
# and these may be replaced in modules by e.g:
# ---------
#     from xtgeo.roxutils._roxar_loader import roxar
#     if TYPE_CHECKING:
#         from xtgeo.grid3d.grid_property import GridProperty
#         if roxar is not None:
#            from roxar.grids import Grid3D as RoxarGridType
# ---------


def load_module(module_name: str) -> Optional[Any]:
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return None


rmsapi = load_module("rmsapi")
roxar = rmsapi if rmsapi else load_module("roxar")

if rmsapi:
    roxar_well_picks = load_module("rmsapi.well_picks")
    roxar_jobs = load_module("rmsapi.jobs")
    roxar_grids = load_module("rmsapi.grids")
else:
    roxar_well_picks = load_module("roxar.well_picks")
    roxar_jobs = load_module("roxar.jobs")
    roxar_grids = load_module("roxar.grids")

# Use TypeVar correctly
RoxarType = TypeVar("RoxarType")
RoxarGridType = TypeVar("RoxarGridType")
RoxarGrid3DType = TypeVar("RoxarGrid3DType")
RoxarWellpicksType = TypeVar("RoxarWellpicksType")
RoxarJobsType = TypeVar("RoxarJobsType")

if TYPE_CHECKING:
    if roxar is not None:
        import roxar.grids as RoxarGridType  # noqa  # type: ignore
        from roxar.grids import Grid3D as RoxarGrid3DType  # noqa  # type: ignore
        from roxar.jobs import Jobs as RoxarJobsType  # noqa  # type: ignore
        from roxar.well_picks import WellPicks as RoxarWellpicksType  # noqa  # type: ignore

# Explicitly type the roxar* as Optional to indicate it may be None
roxar: Optional[Any] = roxar
roxar_grids: Optional[Any] = roxar_grids
roxar_well_picks: Optional[Any] = roxar_well_picks
roxar_jobs: Optional[Any] = roxar_jobs
