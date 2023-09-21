# -*- coding: utf-8 -*-
# flake8: noqa
# pylint: skip-file
# type: ignore

"""The XTGeo Python library."""


import os
import timeit
import warnings

try:
    from ._theversion import version

    __version__ = version
except ImportError:
    __version__ = "0.0.0"


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


# to avoid problems in batch runs when no DISPLAY is set:
_xprint("Import matplotlib etc...")
if not ROXAR:
    import matplotlib as mplib

    display = os.environ.get("DISPLAY", "")
    host1 = os.environ.get("HOSTNAME", "")
    host2 = os.environ.get("HOST", "")
    dhost = host1 + host2 + display

    ertbool = "LSB_JOBID" in os.environ

    if display == "" or "grid" in dhost or "lgc" in dhost or ertbool:
        _xprint("")
        _xprint("=" * 79)

        _xprint(
            "XTGeo info: No display found or a batch (e.g. ERT) server. "
            "Using non-interactive Agg backend for matplotlib"
        )
        mplib.use("Agg")
        _xprint("=" * 79)

#
# Order matters!
#
_xprint("Import matplotlib etc...DONE")

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
from xtgeo.common.xtgeo_dialog import XTGeoDialog
from xtgeo.cxtgeo._cxtgeo import XTGeoCLibError

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

from xtgeo.surface import regular_surface
from xtgeo.surface.regular_surface import RegularSurface
from xtgeo.surface.surfaces import Surfaces

_xprint("Import various XTGeo modules... surface...")

from xtgeo.cube import cube1
from xtgeo.cube.cube1 import Cube

_xprint("Import various XTGeo modules... cube...")

from xtgeo.grid3d import GridRelative, Units, grid, grid_properties, grid_property
from xtgeo.grid3d.grid import Grid
from xtgeo.grid3d.grid_properties import GridProperties, gridproperties_dataframe
from xtgeo.grid3d.grid_property import GridProperty

_xprint("Import various XTGeo modules... 3D grids...")

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

if not ROXAR:
    from xtgeo.plot import baseplot, grid3d_slice, xsection, xtmap

_xprint("Import various XTGeo modules... plots...")

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
from xtgeo.grid3d.grid_properties import gridproperties_from_file
from xtgeo.grid3d.grid_property import gridproperty_from_file, gridproperty_from_roxar
from xtgeo.surface.regular_surface import (
    surface_from_cube,
    surface_from_file,
    surface_from_grid3d,
    surface_from_roxar,
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
