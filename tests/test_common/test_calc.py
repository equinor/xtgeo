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
    """Convert I J K to IB index (F or C order)"""

    ib = xcalc.ijk_to_ib(2, 2, 2, 3, 4, 5)
    assert ib == 16

    ib = xcalc.ijk_to_ib(3, 1, 1, 3, 4, 5)
    assert ib == 2

    ic = xcalc.ijk_to_ib(2, 2, 2, 3, 4, 5, forder=False)
    assert ic == 26

    ic = xcalc.ijk_to_ib(3, 1, 1, 3, 4, 5, forder=False)
    assert ic == 40


def test_ib_to_ijk():
    """Convert IB index to IJK tuple."""

    ijk = xcalc.ib_to_ijk(16, 3, 4, 5)
    assert ijk[0] == 2

    ijk = xcalc.ib_to_ijk(5, 3, 4, 1, forder=True)
    assert ijk == (3, 2, 1)

    ijk = xcalc.ib_to_ijk(5, 3, 4, 1, forder=False)
    assert ijk == (2, 2, 1)

    ijk = xcalc.ib_to_ijk(40, 3, 4, 5, forder=False)
    assert ijk == (3, 1, 1)

    ijk = xcalc.ib_to_ijk(2, 3, 4, 5, forder=True)
    assert ijk == (3, 1, 1)


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


def test_tetrahedron_volume():
    """Compute volume of general tetrahedron"""

    # verfied vs http://tamivox.org/redbear/tetra_calc/index.html

    vert = [0, 0, 0, 2, 1, 0, 0, 2, 0, 0, 0, 2]

    vert = np.array(vert, dtype=np.float64)

    vol = xcalc.tetrehedron_volume(vert)

    assert vol == pytest.approx(1.33333, abs=0.0001)

    vert = [0.9, 111.0, 17, 2, 1, 0, 333, 444, 555, 0, 0, 2]

    assert xcalc.tetrehedron_volume(vert) == pytest.approx(31178.35000000, abs=0.01)

    vert = [0.9, 111.0, 17, 2, 1, 0, 333, 444, 555, 0, 0, 2]
    vert = np.array(vert)
    vert = vert.reshape((4, 3))

    assert xcalc.tetrehedron_volume(vert) == pytest.approx(31178.35000000, abs=0.01)


def test_point_in_tetrahedron():
    """Test if a point is inside a tetrahedron"""

    vert = [0, 0, 0, 2, 1, 0, 0, 2, 0, 0, 0, 2]

    assert xcalc.point_in_tetrahedron(0, 0, 0, vert) is True
    assert xcalc.point_in_tetrahedron(-1, 0, 0, vert) is False
    assert xcalc.point_in_tetrahedron(2, 1, 0, vert) is True
    assert xcalc.point_in_tetrahedron(0, 2, 0, vert) is True
    assert xcalc.point_in_tetrahedron(0, 0, 2, vert) is True


def test_point_in_hexahedron():
    """Test if a point is inside a hexahedron"""

    vrt = [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1]

    vertices = np.array(vrt, dtype=np.float64).reshape(8, 3)

    # assert xcalc.point_in_hexahedron(0, 0, 0, vertices) is True
    assert xcalc.point_in_hexahedron(0.5, 0.5, 0.5, vertices) is True
    assert xcalc.point_in_hexahedron(-0.1, 0.5, 0.5, vertices) is False

    vert = [
        461351.493253,
        5938298.477428,
        1850,
        461501.758690,
        5938385.850231,
        1850,
        461440.718409,
        5938166.753852,
        1850.1,
        461582.200838,
        5938248.702782,
        1850,
        461354.611430,
        5938300.454809,
        1883.246948,
        461504.611754,
        5938387.700867,
        1915.005005,
        461443.842986,
        5938169.007646,
        1904.730957,
        461585.338388,
        5938250.905010,
        1921.021973,
    ]

    assert xcalc.point_in_hexahedron(461467.513586, 5938273.910537, 1850, vert) is True
    assert (
        xcalc.point_in_hexahedron(461467.513586, 5938273.910537, 1849.95, vert)
        is False  # shall be False
    )
    vrt = [
        0.0,
        0.0,
        0.0,
        0.0,
        100.0,
        0.0,
        100.0,
        0.0,
        0.0,
        100.0,
        100.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        100.0,
        1.0,
        100.0,
        0.0,
        1.0,
        100.0,
        100.0,
        1.0,
    ]
    assert xcalc.point_in_hexahedron(0.1, 0.1, 0.1, vrt) is True
