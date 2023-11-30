# flake8: noqa
# pylint: skip-file
# type: ignore

"""The XTGeo Python library."""


import os
import timeit
import warnings


def _timer(*args):
    time1 = timeit.default_timer()

    if args:
        return time1 - args[0]

    return time1


TIME0 = _timer()

DEBUG = 19

if os.environ.get("XTG_DEBUG_DEV") is None:
    DEBUG = 0


def _xprint(msg):
    difftime = _timer(TIME0)

    if DEBUG:
        print(f"({difftime:4.3f})  {msg}")


_xprint("XTGEO __init__ ...")

ROXAR = True
try:
    import roxar
except Exception:
    ROXAR = False

if not ROXAR:
    _display = os.environ.get("DISPLAY", "")
    _hostname = os.environ.get("HOSTNAME", "")
    _host = os.environ.get("HOST", "")

    _dhost = _hostname + _host + _display
    _lsf_job = "LSB_JOBID" in os.environ

    if _display == "" or "grid" in _dhost or "lgc" in _dhost or _lsf_job:
        _xprint("")
        _xprint("=" * 79)
        _xprint(
            "XTGeo info: No display found or a batch (e.g. ERT) server. "
            "Using non-interactive Agg backend for matplotlib"
        )
        _xprint("=" * 79)
        os.environ["MPLBACKEND"] = "Agg"

from xtgeo._cxtgeo import XTGeoCLibError
from xtgeo.common import XTGeoDialog
from xtgeo.common.constants import UNDEF, UNDEF_INT, UNDEF_INT_LIMIT, UNDEF_LIMIT
from xtgeo.common.exceptions import (
    BlockedWellsNotFoundError,
    DateNotFoundError,
    GridNotFoundError,
    KeywordFoundNoDateError,
    KeywordNotFoundError,
    WellNotFoundError,
)
from xtgeo.common.sys import _XTGeoFile

try:
    from xtgeo.common.version import __version__, version
except ImportError:
    __version__ = version = "0.0.0"

_xprint("Import common... done")

_xprint("Import various XTGeo modules...")

from xtgeo.roxutils import roxutils
from xtgeo.roxutils.roxutils import RoxUtils
from xtgeo.well import blocked_well, blocked_wells, well1, wells
from xtgeo.well.blocked_well import BlockedWell
from xtgeo.well.blocked_wells import BlockedWells
from xtgeo.well.well1 import Well
from xtgeo.well.wells import Wells

_xprint("Import various XTGeo modules... wells...")

from xtgeo.grid3d import GridRelative, Units, grid, grid_properties, grid_property
from xtgeo.grid3d.grid import Grid
from xtgeo.grid3d.grid_properties import (
    GridProperties,
    gridproperties_dataframe,
    gridproperties_from_file,
)
from xtgeo.grid3d.grid_property import (
    GridProperty,
    gridproperty_from_file,
    gridproperty_from_roxar,
)

_xprint("Import various XTGeo modules... 3D grids...")

from xtgeo.surface import regular_surface
from xtgeo.surface.regular_surface import (
    RegularSurface,
    surface_from_cube,
    surface_from_file,
    surface_from_grid3d,
    surface_from_roxar,
)
from xtgeo.surface.surfaces import Surfaces

_xprint("Import various XTGeo modules... surface...")

from xtgeo.cube import cube1
from xtgeo.cube.cube1 import Cube

_xprint("Import various XTGeo modules... cube...")


from xtgeo.metadata.metadata import (
    MetaDataCPGeometry,
    MetaDataCPProperty,
    MetaDataRegularCube,
    MetaDataRegularSurface,
    MetaDataWell,
)
from xtgeo.xyz import points, polygons
from xtgeo.xyz.points import Points
from xtgeo.xyz.polygons import Polygons

_xprint("Import various XTGeo modules... xyz...")

_xprint("Import various XTGeo modules...DONE")

# some function wrappers to initiate objects from imports
_xprint("Import various XTGeo wrappers...")
from xtgeo.cube.cube1 import cube_from_file, cube_from_roxar
from xtgeo.grid3d.grid import (
    create_box_grid,
    grid_from_cube,
    grid_from_file,
    grid_from_roxar,
)
from xtgeo.well.blocked_well import blockedwell_from_file, blockedwell_from_roxar
from xtgeo.well.blocked_wells import blockedwells_from_files, blockedwells_from_roxar
from xtgeo.well.well1 import well_from_file, well_from_roxar
from xtgeo.well.wells import wells_from_files
from xtgeo.xyz.points import (
    points_from_file,
    points_from_roxar,
    points_from_surface,
    points_from_wells,
    points_from_wells_dfrac,
)
from xtgeo.xyz.polygons import (
    polygons_from_file,
    polygons_from_roxar,
    polygons_from_wells,
)

warnings.filterwarnings("default", category=DeprecationWarning, module="xtgeo")

_xprint("XTGEO __init__ done")
