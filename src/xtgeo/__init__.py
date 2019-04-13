# -*- coding: utf-8 -*-
# flake8: noqa
# pylint: skip-file
"""The XTGeo Python module."""

from __future__ import division, absolute_import
from __future__ import print_function
import timeit


def timer(*args):
    """Without args; return the time, with a time as arg return the
    difference.
    """
    time1 = timeit.default_timer()

    if args:
        return time1 - args[0]

    return time1

TIME0 = timer()
DEBUG = 1

def xprint(msg):

    difftime = timer(TIME0)

    if DEBUG:
        print('>>>>>>>>>>>>> ({0:4.3f})  {1}'.format(difftime, msg))

xprint('XTGEO __init__ ...')

import os
# to avoid problems in batch runs when no DISPLAY is set:
xprint('Import matplotlib etc...')
import matplotlib as mplib

display = os.environ.get('DISPLAY', '')
host1 = os.environ.get('HOSTNAME', '')
host2 = os.environ.get('HOST', '')
dhost = host1 + host2 + display

ertbool = 'LSB_JOBID' in os.environ

if display == '' or 'grid' in dhost or 'lgc' in dhost or ertbool:

    print('')
    print('=' * 79)

    print(
        'XTGeo info: No display found or ERT server. Using non-interactive '
        'Agg backend for matplotlib'
    )
    mplib.use('Agg')
    print('=' * 79)

#
# Order matters!
#
xprint('Import matplotlib etc...DONE')

xprint('Import xtgeo version...')
from xtgeo._theversion import theversion

__version__ = theversion()

xprint('Import xtgeo version...DONE')

xprint('Import common...')

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

xprint('Import common... done')

xprint('Import various XTGeo modules...')

from xtgeo.roxutils import roxutils
from xtgeo.roxutils.roxutils import RoxUtils

from xtgeo.well import well
from xtgeo.well import wells
from xtgeo.well import blocked_well
from xtgeo.well import blocked_wells

from xtgeo.surface import regular_surface

from xtgeo.cube import cube

from xtgeo.grid3d import grid
from xtgeo.grid3d import grid_property
from xtgeo.grid3d import grid_properties

from xtgeo.xyz import points
from xtgeo.xyz import polygons

from xtgeo.plot import baseplot
from xtgeo.plot import xsection
from xtgeo.plot import xtmap
from xtgeo.plot import grid3d_slice

xprint('Import various XTGeo modules...DONE')


# some function wrappers to initiate objects from imports
xprint('Import various XTGeo wrappers...')
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

xprint('XTGEO __init__ done')
