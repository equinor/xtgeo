# -*- coding: utf-8 -*-

import math

import numpy as np
import pytest

import xtgeo
import xtgeo.common.calc as xcalc
import xtgeo.cxtgeo._cxtgeo as _cxtgeo

xtg = xtgeo.XTGeoDialog()
from xtgeo.common import logger
from xtgeo.common.xtgeo_dialog import testdatafolder

TPATH = testdatafolder

# =============================================================================
# Do tests of simple calc routines
# =============================================================================
TESTGRID = TPATH / "3dgrids/etc/gridqc1.roff"
TESTGRID_TBULK = TPATH / "3dgrids/etc/gridqc1_totbulk.roff"
TESTGRID2 = TPATH / "3dgrids/etc/banal6.roff"
TESTGRID3 = TPATH / "3dgrids/etc/box.roff"
TESTGRID4 = TPATH / "3dgrids/etc/twocell.roff"


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


@pytest.mark.parametrize(
    "iin, jin, xcor, ycor, xinc, yinc, ncol, nrow, yflip, rota, exp_xori, exp_yori",
    [
        (0, 0, 0.0, 0.0, 50.0, 50.0, 2, 2, 1, 0.0, 0.0, 0.0),
        (0, 0, 0.0, 0.0, 50.0, 50.0, 2, 2, 1, 90.0, 0.0, 0.0),
        (0, 0, 100.0, 300.0, 50.0, 50.0, 2, 2, 1, 0.0, 100.0, 300.0),
        (1, 1, 100.0, 300.0, 50.0, 50.0, 2, 2, 1, 0.0, 50.0, 250.0),
        (1, 1, 100.0, 300.0, 50.0, 50.0, 2, 2, 1, 360.0, 50.0, 250.0),
        (1, 1, 100.0, 300.0, 50.0, 50.0, 2, 2, 1, 90.0, 150.0, 250.0),
        (1, 1, 100.0, 300.0, 50.0, 50.0, 2, 2, 1, 180.0, 150.0, 350.0),
        (2, 2, 100.0, 300.0, 50.0, 50.0, 3, 3, 1, 0.0, 0.0, 200.0),
        (2, 2, 100.0, 300.0, 50.0, 100.0, 3, 3, 1, 0.0, 0.0, 100.0),
        (2, 2, 100.0, 300.0, 50.0, 100.0, 3, 3, 1, 360.0, 0.0, 100.0),
        (2, 2, 100.0, 300.0, 50.0, 100.0, 3, 3, 1, 720.0, 0.0, 100.0),
        (0, 2, 0.0, 0.0, 50.0, 50.0, 3, 3, 1, 45.0, 70.7107, -70.7107),
        (0, 2, 0.0, 0.0, 50.0, 50.0, 3, 3, 1, 225.0, -70.7107, 70.7107),
        (2, 2, 0.0, 0.0, 100.0, 50.0, 3, 3, 1, 30.0, -123.2051, -186.6025),
        (2, 0, 0.0, 0.0, 50.0, 50.0, 3, 3, 1, 45.0, -70.7107, -70.7107),
    ],
)
def test_xyori_from_ij(
    iin,
    jin,
    xcor,
    ycor,
    xinc,
    yinc,
    ncol,
    nrow,
    yflip,
    rota,
    exp_xori,
    exp_yori,
):
    """Testing correct origin given a XY point with indices and geometrics."""
    xori, yori = xcalc.xyori_from_ij(
        iin, jin, xcor, ycor, xinc, yinc, ncol, nrow, yflip, rota
    )

    assert xori == pytest.approx(exp_xori, rel=0.001)
    assert yori == pytest.approx(exp_yori, rel=0.001)


@pytest.mark.parametrize(
    "iin, jin, xcor, ycor, xinc, yinc, ncol, nrow, yflip, rota",
    [
        (3, 3, 100.0, 300.0, 50.0, 100.0, 3, 3, 1, 0.0),
        (-1, 3, 100.0, 300.0, 50.0, 100.0, 3, 3, 1, 0.0),
    ],
)
def test_xyori_from_ij_fails(
    iin,
    jin,
    xcor,
    ycor,
    xinc,
    yinc,
    ncol,
    nrow,
    yflip,
    rota,
):
    """Correct origin given a XY point with indices and geometrics will fail."""

    with pytest.raises(ValueError):
        _, _ = xcalc.xyori_from_ij(
            iin, jin, xcor, ycor, xinc, yinc, ncol, nrow, yflip, rota
        )
    return


def test_ijk_to_ib():
    """Convert I J K to IB index (F or C order)."""

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


def test_vectorpair_angle3d():
    """Testing getting angles from vectors in 3D given by 3 XYZ points"""

    # checks via https://onlinemschool.com/math/assistance/vector/angl/

    angle1 = xcalc.vectorpair_angle3d((0, 0, 0), (1, 1, 0), (0, 4, 0))

    assert angle1 == pytest.approx(45.0)

    angle2 = xcalc.vectorpair_angle3d((0, 0, 0), (-10, 1, 5), (0, 40, -9))

    assert angle2 == pytest.approx(90.6225)

    angle3 = xcalc.vectorpair_angle3d((0, 0, 0), (-10, 1, 0), (0, 40, 0))

    assert angle3 == pytest.approx(84.2894)

    angle4 = xcalc.vectorpair_angle3d(
        (0, 0, 0), (-10, 1, 5), (0, 40, -9), birdview=True
    )

    assert angle4 == pytest.approx(angle3)

    angle1 = xcalc.vectorpair_angle3d((0, 0, 0), (1, 1, 0), (0, 0, 0))
    assert angle1 is None

    with pytest.raises(ValueError):
        angle1 = xcalc.vectorpair_angle3d((0, 0, 0, 99), (1, 1, 0), (0, 4, 0))


def test_x_cellangles():
    """Test x_minmax_cellangles* functions (lowlevel call)"""

    grd = xtgeo.grid_from_file(TESTGRID)
    cell1 = grd.get_xyz_cell_corners((6, 4, 1))
    cell2 = grd.get_xyz_cell_corners((6, 3, 1))
    cell3 = grd.get_xyz_cell_corners((4, 7, 1))

    _ier, amin, amax = _cxtgeo.x_minmax_cellangles_topbase(cell1, 0, 1)

    assert amin == pytest.approx(71.329, abs=0.01)
    assert amax == pytest.approx(102.673, abs=0.01)

    # birdview
    _ier, amin, amax = _cxtgeo.x_minmax_cellangles_topbase(cell1, 1, 1)

    assert amin == pytest.approx(89.701, abs=0.01)
    assert amax == pytest.approx(90.274, abs=0.01)

    # side cells
    _ier, amin, amax = _cxtgeo.x_minmax_cellangles_sides(cell1, 1)

    _ier, amin, amax = _cxtgeo.x_minmax_cellangles_sides(cell2, 1)

    assert amin == pytest.approx(49.231, abs=0.01)
    assert amax == pytest.approx(130.77, abs=0.01)

    _ier, amin, amax = _cxtgeo.x_minmax_cellangles_sides(cell3, 1)

    assert amin == pytest.approx(75.05, abs=0.01)
    assert amax == pytest.approx(104.95, abs=0.01)


def test_get_cell_volume():
    """Test hexahedron (cell) bulk volume valculation"""

    # box
    grd = xtgeo.grid_from_file(TESTGRID3)

    vol1 = grd.get_cell_volume((1, 1, 1))
    assert vol1 == pytest.approx(3821600, rel=0.01)

    # banal6
    grd = xtgeo.grid_from_file(TESTGRID2)

    vol1 = grd.get_cell_volume((1, 1, 1))
    vol2 = grd.get_cell_volume((4, 1, 1))
    vol3 = grd.get_cell_volume((1, 2, 1))
    vol4 = grd.get_cell_volume((3, 1, 2))

    assert vol1 == pytest.approx(1679.7, rel=0.01)
    assert vol2 == pytest.approx(2070.3, rel=0.01)
    assert vol3 == pytest.approx(1289.1, rel=0.01)
    assert vol4 == pytest.approx(593.75, rel=0.01)

    # gridqc1
    grd = xtgeo.grid_from_file(TESTGRID)
    tbulk_rms = xtgeo.gridproperty_from_file(TESTGRID_TBULK)

    rmean = []
    for prec in [1, 2, 4]:
        ntot = 0
        nfail = 0
        ratioarr = []
        for icol in range(grd.ncol):
            for jrow in range(grd.nrow):
                for klay in range(grd.nlay):
                    vol1a = grd.get_cell_volume(
                        (icol, jrow, klay), zerobased=True, precision=prec
                    )
                    if vol1a is not None:
                        vol1b = tbulk_rms.values[icol, jrow, klay]
                        ratio = vol1a / vol1b
                        ratioarr.append(ratio)
                        ntot += 1
                        if ratio < 0.98 or ratio > 1.02:
                            nfail += 1
                            logger.info("%s %s %s:  %s", icol, jrow, klay, ratio)
                            logger.info("XTGeo vs RMS %s %s", vol1a, vol1b)
                        if prec > 1:
                            assert vol1a == pytest.approx(vol1b, 0.0001)

        rarr = np.array(ratioarr)
        rmean.append(rarr.mean())
        logger.info(
            "Prec: %s, Fails of total %s vs %s, mean/min/max: %s %s %s",
            prec,
            nfail,
            ntot,
            rarr.mean(),
            rarr.min(),
            rarr.max(),
        )
        if prec > 1:
            assert rarr == pytest.approx(1.0, 0.0001)
            assert nfail == 0

    # ensure that mean difference get closer to 1 with increasing precision?
    for ravg in rmean:
        diff = abs(1.0 - ravg)
        logger.info("Diff from 1: %s", diff)
