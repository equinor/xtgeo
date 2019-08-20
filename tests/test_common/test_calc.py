# -*- coding: utf-8 -*-
import math
import xtgeo.common.calc as xcalc


# =============================================================================
# Do tests of simple calc routines
# =============================================================================


def test_ijk_to_ib():
    """Convert I J K to IB index."""

    ib = xcalc.ijk_to_ib(2, 2, 2, 3, 4, 5)
    assert ib == 16


def test_ib_to_ijk():
    """Convert IB index to IJK tuple."""

    ijk = xcalc.ib_to_ijk(16, 3, 4, 5)
    assert ijk[0] == 2


def test_angle2azimuth():
    """Test angle to azimuth conversion"""

    res = xcalc.angle2azimuth(30)
    assert res == 60.0

    a1 = 30 * math.pi / 180
    a2 = 60 * math.pi / 180

    res = xcalc.angle2azimuth(a1, mode="radians")
    assert res == a2

    res = xcalc.angle2azimuth(-30)
    assert res == 120.0
