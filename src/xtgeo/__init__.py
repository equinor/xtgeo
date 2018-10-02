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
from xtgeo.plot import baseplot
from xtgeo.plot import xsection
from xtgeo.plot import xtmap
from xtgeo.plot import grid3d_slice
from xtgeo.xyz import points
from xtgeo.xyz import polygons

from xtgeo.common.constants import UNDEF
from xtgeo.common.constants import UNDEF_LIMIT
from xtgeo.common.constants import UNDEF_INT
from xtgeo.common.constants import UNDEF_INT_LIMIT

# from xtgeo.xyz import _xyz

from xtgeo._version import get_versions
__version__ = get_versions()['version']
del get_versions

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
