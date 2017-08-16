"""The XTGeo Python module."""
import os

# to avoid problems in batch runs when no DISPLAY is set:
import matplotlib as mplib
display = os.environ.get('DISPLAY', '')
host1 = os.environ.get('HOSTNAME', '')
host2 = os.environ.get('HOST', '')
host = host1 + host2

print('')
print('=' * 79)
if display == '' or 'grid' in host1 or 'lgc' in host1 or\
   'LSB_JOBID' in os.environ:

    print('XTGeo info: No display found or ERT server. Using non-interactive '
          'Agg backend for matplotlib')
    mplib.use('Agg')
else:
    print('XTGeo info: Matplotlib Agg backend is NOT active, '
          'allow interactive plots')

print('=' * 79)

from .surface import regular_surface
from .grid3d import grid
from .grid3d import grid_property
from .grid3d import grid_properties
from .well import well

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
