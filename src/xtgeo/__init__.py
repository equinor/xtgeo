# flake8: noqa
"""The XTGeo Python module."""
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


from .surface import regular_surface
from .cube import cube
from .grid3d import grid
from .grid3d import grid_property
from .grid3d import grid_properties
from .well import well
from .plot import baseplot
from .plot import xsection
from .plot import xtmap
from .plot import grid3d_slice
from .xyz import points
from .xyz import polygons
# from .xyz import xyz

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

# some function wrappers to initiate objects from imports
from .surface.regular_surface import surface_from_file
from .surface.regular_surface import surface_from_roxar
from .grid3d.grid_property import gridproperty_from_file
from .grid3d.grid_property import gridproperty_from_roxar

from .cube.cube import cube_from_file
from .cube.cube import cube_from_roxar
