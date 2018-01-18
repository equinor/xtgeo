# -*- coding: utf-8 -*-
import os
import warnings
import pytest

import xtgeo.common.calc as xcalc
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

td = xtg.tmpdir
testpath = xtg.testpath


# =============================================================================
# Some useful functions
# =============================================================================

def assert_equal(this, that, txt=''):
    assert this == that, txt


def assert_almostequal(this, that, tol, txt=''):
    assert this == pytest.approx(that, abs=tol), txt


# SEGYIO ----------------------------------------------------------------------
no_segyio = False
try:
    import segyio  # pylint: disable=F401 # noqa:<Error No>
except ImportError:
    no_segyio = True

if no_segyio:
    warnings.warn('"segyio" library not found')

skipsegyio = pytest.mark.skipif(no_segyio, reason='Skip test with segyio')

# Roxar python-----------------------------------------------------------------
# Routines using matplotlib shall not ran if ROXENV=1
# use the @skipifroxar decorator

roxar = False
if 'ROXENV' in os.environ:
    roxenv = str(os.environ.get('ROXENV'))
    roxar = True
    print(roxenv)
    warnings.warn('Roxar is present')


skipifroxar = pytest.mark.skipif(roxar, reason='Skip test in Roxar python')

skipunlessroxar = pytest.mark.skipif(not roxar,
                                     reason='Skip if NOT Roxar python')

# =============================================================================
# Do tests
# =============================================================================


def test_ijk_to_ib():
    """Convert I J K to IB index."""

    ib = xcalc.ijk_to_ib(2, 2, 2, 3, 4, 5)
    logger.info(ib)
    assert ib == 16


def test_ib_to_ijk():
    """Convert IB index to IJK tuple."""

    ijk = xcalc.ib_to_ijk(16, 3, 4, 5)
    logger.info(ijk)
    assert ijk[0] == 2
