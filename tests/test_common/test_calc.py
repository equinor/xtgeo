# -*- coding: utf-8 -*-

import math
import pytest
import numpy as np

import xtgeo.common.calc as xcalc


# =============================================================================
# Do tests of simple calc routines
# =============================================================================

def test_vectorinfo2():
    """Testing vectorinfo2 function"""
    llen, rad, deg = xcalc.vectorinfo2(0, 10, 0, 10, option=1)
    assert llen == pytest.approx(math.sqrt(100 + 100), 0.001)
    assert rad == pytest.approx(math.pi / 4.0, 0.001)
    assert deg == pytest.approx(45, 0.001)

    llen, rad, deg = xcalc.vectorinfo2(0, 10, 0, 0, option=0)
    assert llen == pytest.approx(10.0, 0.001)
    assert rad == pytest.approx(math.pi / 2.0, 0.001)
    assert deg == pytest.approx(90, 0.001)


def test_diffangle():
    """Testing difference between two angles"""

    assert xcalc.diffangle(30, 40) == -10.0
    assert xcalc.diffangle(10, 350) == 20.0
    assert xcalc.diffangle(360, 340) == 20.0
    assert xcalc.diffangle(360, 170) == -170.0


def test_ijk_to_ib():
    """Convert I J K to IB index."""

    ib = xcalc.ijk_to_ib(2, 2, 2, 3, 4, 5)
    assert ib == 16


def test_ib_to_ijk():
    """Convert IB index to IJK tuple."""

    ijk = xcalc.ib_to_ijk(16, 3, 4, 5)
    assert ijk[0] == 2

    ijk = xcalc.ib_to_ijk(5, 3, 4, 1, forder=True)
    assert ijk == (3, 2, 1)

    ijk = xcalc.ib_to_ijk(5, 3, 4, 1, forder=False)
    assert ijk == (2, 2, 1)


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

    res = xcalc.angle2azimuth(-300)
    assert res == 30


def test_azimuth2angle():
    """Test azimuth to school angle conversion"""

    res = xcalc.azimuth2angle(60)
    assert res == 30.0

    a1 = 60 * math.pi / 180
    a2 = 30 * math.pi / 180

    res = xcalc.azimuth2angle(a1, mode="radians")
    assert res == a2

    res = xcalc.azimuth2angle(120)
    assert res == 330

    res = xcalc.azimuth2angle(-30)
    assert res == 120


def test_averageangle():
    """Test finding an average angle"""

    # input can be a list...
    assert xcalc.averageangle([355, 5]) == 0.0

    # input can be a tuple...
    assert xcalc.averageangle((355, 5)) == 0.0

    # input can be a numpy array
    assert xcalc.averageangle(np.array([355, 5])) == 0.0

    # input with many elements
    assert xcalc.averageangle([340, 5, 355, 20]) == 0.0
