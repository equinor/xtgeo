# ruff: noqa
# type: ignore
"""The XTGeo Python library."""

import os
import platform
import sys
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


try:
    from xtgeo.common.version import __version__, version
except ImportError:
    __version__ = version = "0.0.0"

from xtgeo._cxtgeo import XTGeoCLibError
from xtgeo.common import XTGeoDialog
from xtgeo.common.constants import UNDEF, UNDEF_INT, UNDEF_INT_LIMIT, UNDEF_LIMIT
from xtgeo.common.exceptions import (
    BlockedWellsNotFoundError,
    DateNotFoundError,
    GridNotFoundError,
    InvalidFileFormatError,
    KeywordFoundNoDateError,
    KeywordNotFoundError,
    WellNotFoundError,
)

_xprint("Import common... done")

_xprint("Import various XTGeo modules...")

from xtgeo.metadata.metadata import (
    MetaDataCPGeometry,
    MetaDataCPProperty,
    MetaDataRegularCube,
    MetaDataRegularSurface,
    MetaDataWell,
)

_xprint("Import various XTGeo modules... metadata...")

from xtgeo.roxutils import roxutils
from xtgeo.roxutils.roxutils import RoxUtils

from xtgeo.well import blocked_well, blocked_wells, well1, wells
from xtgeo.well.blocked_well import (
    BlockedWell,
    blockedwell_from_file,
    blockedwell_from_roxar,
)
from xtgeo.well.blocked_wells import (
    BlockedWells,
    blockedwells_from_files,
    blockedwells_from_roxar,
)
from xtgeo.well.well1 import Well, well_from_file, well_from_roxar
from xtgeo.well.wells import Wells, wells_from_files
from xtgeo.xyz.points import (
    points_from_file,
    points_from_roxar,
    points_from_surface,
    points_from_wells,
    points_from_wells_dfrac,
)

_xprint("Import various XTGeo modules... wells...")

from xtgeo.grid3d._ecl_grid import GridRelative, Units
from xtgeo.grid3d.grid import Grid
from xtgeo.grid3d.grid_properties import (
    GridProperties,
    gridproperties_dataframe,
    gridproperties_from_file,
    list_gridproperties,
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
from xtgeo.xyz.polygons import (
    polygons_from_file,
    polygons_from_roxar,
    polygons_from_wells,
)

warnings.filterwarnings("default", category=DeprecationWarning, module="xtgeo")

_xprint("XTGEO __init__ done")

# Remove symbols imported for internal use
del os, platform, sys, timeit, warnings, TIME0, DEBUG, _timer, _xprint

# Let type-checkers know what is exported
__all__ = [
    "BlockedWell",
    "BlockedWells",
    "BlockedWellsNotFoundError",
    "Cube",
    "DateNotFoundError",
    "Grid",
    "GridNotFoundError",
    "GridProperties",
    "GridProperty",
    "GridRelative",
    "GridRelative",
    "InvalidFileFormatError",
    "KeywordFoundNoDateError",
    "KeywordNotFoundError",
    "MetaDataCPGeometry",
    "MetaDataCPProperty",
    "MetaDataRegularCube",
    "MetaDataRegularSurface",
    "MetaDataWell",
    "Points",
    "Polygons",
    "RegularSurface",
    "RoxUtils",
    "Surfaces",
    "UNDEF",
    "UNDEF_INT",
    "UNDEF_INT_LIMIT",
    "UNDEF_LIMIT",
    "Units",
    "Units",
    "Well",
    "WellNotFoundError",
    "Wells",
    "XTGeoCLibError",
    "XTGeoDialog",
    "__version__",
    "blocked_well",
    "blocked_wells",
    "blockedwell_from_file",
    "blockedwell_from_roxar",
    "blockedwells_from_file",
    "blockedwells_from_files",
    "blockedwells_from_roxar",
    "create_box_grid",
    "cube1",
    "cube_from_file",
    "cube_from_roxar",
    "grid",
    "grid",
    "grid_from_cube",
    "grid_from_file",
    "grid_from_roxar",
    "grid_properties",
    "grid_properties",
    "grid_property",
    "grid_property",
    "gridproperties_dataframe",
    "gridproperties_from_file",
    "gridproperty_from_file",
    "gridproperty_from_roxar",
    "list_gridproperties",
    "points",
    "points_from_file",
    "points_from_roxar",
    "points_from_surface",
    "points_from_wells",
    "points_from_wells_dfrac",
    "polygons",
    "polygons_from_file",
    "polygons_from_roxar",
    "polygons_from_wells",
    "regular_surface",
    "roxutils",
    "surface_from_cube",
    "surface_from_file",
    "surface_from_grid3d",
    "surface_from_roxar",
    "version",
    "well1",
    "well_from_file",
    "well_from_roxar",
    "wells",
    "wells_from_files",
]
