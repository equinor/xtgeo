# -*- coding: utf-8 -*-
# flake8: noqa
# pylint: skip-file
"""The XTGeo Python module."""

from __future__ import division, absolute_import
from __future__ import print_function
import os
import timeit

try:
    from ._theversion import version
    __version__ = version
except ImportError:
    __version__ = '0.0.0'

def _timer(*args):
    time1 = timeit.default_timer()

    if args:
        return time1 - args[0]

    return time1

TIME0 = _timer()

DEBUG = 19

if os.environ.get("XTG_VERBOSE_LEVEL") is None:
    DEBUG = 0

def _xprint(msg):

    difftime = _timer(TIME0)

    if DEBUG:
        print('({0:4.3f})  {1}'.format(difftime, msg))

_xprint('XTGEO __init__ ...')

# to avoid problems in batch runs when no DISPLAY is set:
_xprint('Import matplotlib etc...')
import matplotlib as mplib

display = os.environ.get('DISPLAY', '')
host1 = os.environ.get('HOSTNAME', '')
host2 = os.environ.get('HOST', '')
dhost = host1 + host2 + display

ertbool = 'LSB_JOBID' in os.environ

if display == '' or 'grid' in dhost or 'lgc' in dhost or ertbool:

    _xprint('')
    _xprint('=' * 79)

    _xprint(
        'XTGeo info: No display found or a batch (e.g. ERT) server. '
        'Using non-interactive Agg backend for matplotlib'
    )
    mplib.use('Agg')
    _xprint('=' * 79)

#
# Order matters!
#
_xprint('Import matplotlib etc...DONE')

from xtgeo.common.constants import UNDEF
from xtgeo.common.constants import UNDEF_LIMIT
from xtgeo.common.constants import UNDEF_INT
from xtgeo.common.constants import UNDEF_INT_LIMIT

from xtgeo.common.exceptions import DateNotFoundError
from xtgeo.common.exceptions import KeywordNotFoundError
from xtgeo.common.exceptions import KeywordFoundNoDateError
from xtgeo.common.exceptions import WellNotFoundError
from xtgeo.common.exceptions import GridNotFoundError
from xtgeo.common.exceptions import BlockedWellsNotFoundError

_xprint('Import common... done')

_xprint('Import various XTGeo modules...')

from xtgeo.roxutils import roxutils
from xtgeo.roxutils.roxutils import RoxUtils

from xtgeo.well import well
from xtgeo.well import wells
from xtgeo.well import blocked_well
from xtgeo.well import blocked_wells
_xprint('Import various XTGeo modules... wells...')

from xtgeo.surface import regular_surface
_xprint('Import various XTGeo modules... surface...')

from xtgeo.cube import cube
_xprint('Import various XTGeo modules... cube...')

from xtgeo.grid3d import grid
from xtgeo.grid3d import grid_property
from xtgeo.grid3d import grid_properties
_xprint('Import various XTGeo modules... 3D grids...')

from xtgeo.xyz import points
from xtgeo.xyz import polygons
_xprint('Import various XTGeo modules... xyz...')


from xtgeo.plot import baseplot
from xtgeo.plot import xsection
from xtgeo.plot import xtmap
from xtgeo.plot import grid3d_slice
_xprint('Import various XTGeo modules... plots...')


_xprint('Import various XTGeo modules...DONE')


# some function wrappers to initiate objects from imports
_xprint('Import various XTGeo wrappers...')
from xtgeo.surface.regular_surface import surface_from_file
from xtgeo.surface.regular_surface import surface_from_roxar
from xtgeo.surface.regular_surface import surface_from_cube

from xtgeo.grid3d.grid import grid_from_file
from xtgeo.grid3d.grid import grid_from_roxar

from xtgeo.grid3d.grid_property import gridproperty_from_file
from xtgeo.grid3d.grid_property import gridproperty_from_roxar

from xtgeo.cube.cube import cube_from_file
from xtgeo.cube.cube import cube_from_roxar

from xtgeo.well.well import well_from_file
from xtgeo.well.well import well_from_roxar

from xtgeo.well.blocked_well import blockedwell_from_file
from xtgeo.well.blocked_well import blockedwell_from_roxar

from xtgeo.well.blocked_wells import blockedwells_from_roxar

from xtgeo.xyz.polygons import polygons_from_file
from xtgeo.xyz.polygons import polygons_from_roxar

from xtgeo.xyz.points import points_from_file
from xtgeo.xyz.points import points_from_roxar

_xprint('XTGEO __init__ done')
