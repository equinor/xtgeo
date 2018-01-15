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
from .grid3d import grid
from .grid3d import grid_property
from .grid3d import grid_properties
from .well import well
from .plot import baseplot
from .plot import xsection
from .xyz import points
from .xyz import polygons

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
