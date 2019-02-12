# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function

import pytest
import os

from xtgeo import RoxUtils
from xtgeo.common import XTGeoDialog
import test_common.test_xtg as tsetup

roxver = None
try:
    import roxar
    roxver = roxar.__version__
except ImportError:
    pass
xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

td = xtg.tmpdir
testpath = xtg.testpath

# =============================================================================
# Do tests
# =============================================================================

PROJ = {}
PROJ['1.1'] = '../xtgeo-testdata-equinor/data/rmsprojects/reek.rms10.1.1'
PROJ['1.1.1'] = '../xtgeo-testdata-equinor/data/rmsprojects/reek.rms10.1.3'
PROJ['1.2.1'] = '../xtgeo-testdata-equinor/data/rmsprojects/reek.rms11.0.1'
PROJ['1.3'] = '../xtgeo-testdata-equinor/data/rmsprojects/reek.rms11.1.0'


@pytest.fixture()
def load_reek():
    """Fixture for loading Reek project"""

    print(roxver)
    if not os.path.isdir(PROJ[roxver]):
        raise RuntimeError('RMS test project is missing for roxar version {}'
                           .format(roxver))
    return RoxUtils(PROJ[roxver])


@tsetup.equinor
@tsetup.skipunlessroxar
def test_basic_api(load_reek):
    """Test some basic API features such as """

    rox = load_reek

    print(rox)
