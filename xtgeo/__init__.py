"""The XTGeo Python module."""
import os

# to avoid problems in batch runs when no DISPLAY is set:
import matplotlib as mplib
if os.environ.get('DISPLAY','') == '':
    print('No display found. Using non-interactive Agg backend for matplotlib')
    mplib.use('Agg')

from .surface import regular_surface
from .grid3d import grid
from .grid3d import grid_property
from .grid3d import grid_properties
from .well import well

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
