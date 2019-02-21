# -*- coding: utf-8 -*-
# flake8: noqa
"""The XTGeo Python module."""

from __future__ import division, absolute_import
from __future__ import print_function

import os

# to avoid problems in batch runs when no DISPLAY is set:
import matplotlib as mplib
display = os.environ.get('DISPLAY', '')
host1 = os.environ.get('HOSTNAME', '')
host2 = os.environ.get('HOST', '')
dhost = host1 + host2 + display

ertbool = 'LSB_JOBID' in os.environ

if display == '' or 'grid' in dhost or 'lgc' in dhost or ertbool:

    print('')
    print('=' * 79)

    print('XTGeo info: No display found or ERT server. Using non-interactive '
          'Agg backend for matplotlib')
    mplib.use('Agg')
    print('=' * 79)


from xtgeo.surface import regular_surface
from xtgeo.cube import cube
from xtgeo.grid3d import grid
from xtgeo.grid3d import grid_property
from xtgeo.grid3d import grid_properties
from xtgeo.well import well
from xtgeo.well import wells
from xtgeo.plot import baseplot
from xtgeo.plot import xsection
from xtgeo.plot import xtmap
from xtgeo.plot import grid3d_slice
from xtgeo.xyz import points
from xtgeo.xyz import polygons
from xtgeo.roxutils import roxutils

from xtgeo.surface.regular_surface import RegularSurface
from xtgeo.cube.cube import Cube
from xtgeo.grid3d.grid import Grid
from xtgeo.grid3d.grid_property import GridProperty
from xtgeo.grid3d.grid_properties import GridProperties
from xtgeo.well.well import Well
from xtgeo.well.wells import Wells
from xtgeo.xyz.points import Points
from xtgeo.xyz.polygons import Polygons
from xtgeo.roxutils.roxutils import RoxUtils

from xtgeo.common.constants import UNDEF
from xtgeo.common.constants import UNDEF_LIMIT
from xtgeo.common.constants import UNDEF_INT
from xtgeo.common.constants import UNDEF_INT_LIMIT

from xtgeo.common.xtgeo_dialog import XTGeoDialog

# from xtgeo.xyz import _xyz

# from xtgeo._version import get_versions
# __version__ = get_versions()['version']
# del get_versions

from xtgeo._theversion import theversion
__version__ = theversion()
# del get_versions

# some function wrappers to initiate objects from imports
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

from xtgeo.xyz.polygons import polygons_from_file
from xtgeo.xyz.polygons import polygons_from_roxar

from xtgeo.xyz.points import points_from_file
from xtgeo.xyz.points import points_from_roxar
