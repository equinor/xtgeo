# -*- coding: utf-8 -*-
import sys
import pytest

import xtgeo.common.calc as xcalc
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg._testsetup():
    sys.exit(-9)

td = xtg.tmpdir
testpath = xtg.testpath


# =============================================================================
# Some useful functions
# =============================================================================

def assert_equal(this, that, txt=''):
    assert this == that, txt


def assert_almostequal(this, that, tol, txt=''):
    assert this == pytest.approx(that, abs=tol), txt


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
