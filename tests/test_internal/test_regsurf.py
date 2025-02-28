"""Test some basic _internal functions which are in C++ and use the pybind11 method.

Some basic methods are tested here, while the more complex ones are tested in an
integrated manner in the other more general tests (like testing surfaces, cubes,
3D grids)

This module focus on testing the C++ functions that are used in the "regsurf"
name space.

"""

import os

import numpy as np
import pytest

import xtgeo
import xtgeo._internal as _internal  # type: ignore
from xtgeo.common.log import functimer, null_logger

logger = null_logger(__name__)


def test_find_cell_range_simple_norotated():
    surf = xtgeo.RegularSurface(
        xori=0, yori=0, xinc=1, yinc=2, ncol=3, nrow=4, rotation=0
    )

    assert isinstance(surf, xtgeo.RegularSurface)

    xmin, xmax = surf.xmin, surf.xmax
    ymin, ymax = surf.ymin, surf.ymax

    print(xmin, xmax, ymin, ymax)
    print(surf.rotation)

    regsurf_cpp = _internal.regsurf.RegularSurface(surf)

    result = regsurf_cpp.find_cell_range(xmin, xmax, ymin, ymax, 0)

    assert result == (0, 2, 0, 3)


def test_find_cell_range_simple_rotated1():
    surf = xtgeo.RegularSurface(
        xori=0, yori=0, xinc=1, yinc=1, ncol=3, nrow=4, rotation=45
    )

    assert isinstance(surf, xtgeo.RegularSurface)

    xmin, xmax = surf.xmin, surf.xmax
    ymin, ymax = surf.ymin, surf.ymax

    print(xmin, xmax, ymin, ymax)
    print(surf.rotation)

    regsurf_cpp = _internal.regsurf.RegularSurface(surf)

    result = regsurf_cpp.find_cell_range(
        xmin,
        xmax,
        ymin,
        ymax,
        0,
    )

    assert result == (0, 2, 0, 3)


def test_find_cell_range_simple_rotated2():
    surf = xtgeo.RegularSurface(
        xori=1000, yori=2000, xinc=1, yinc=1, ncol=6, nrow=5, rotation=30
    )

    xmin, xmax = 1000 - 0.5, 1000 + 0.2
    ymin, ymax = 2000 + 4.2, 2000 + 4.5

    regsurf_cpp = _internal.regsurf.RegularSurface(surf)

    result = regsurf_cpp.find_cell_range(xmin, xmax, ymin, ymax, 0)
    assert result == (2, 2, 4, 4)

    result = regsurf_cpp.find_cell_range(xmin, xmax, ymin, ymax, 1)
    assert result == (1, 3, 3, 4)


def test_find_cell_range_simple_outside():
    surf = xtgeo.RegularSurface(
        xori=0, yori=0, xinc=1, yinc=1, ncol=3, nrow=4, rotation=45
    )

    assert isinstance(surf, xtgeo.RegularSurface)

    xmin, xmax = 1000, 2000
    ymin, ymax = 99, 1001

    regsurf_cpp = _internal.regsurf.RegularSurface(surf)

    result = regsurf_cpp.find_cell_range(xmin, xmax, ymin, ymax, 0)

    assert result == (2, 2, 0, 1)


def test_get_outer_corners():
    surf = xtgeo.RegularSurface(
        xori=0, yori=0, xinc=1, yinc=1, ncol=3, nrow=4, rotation=30
    )
    regsurf_cpp = _internal.regsurf.RegularSurface(surf)

    result = regsurf_cpp.get_outer_corners()

    assert result[0].x == pytest.approx(0.0)
    assert result[0].y == pytest.approx(0.0)
    assert result[1].x == pytest.approx(2.59808, rel=0.01)
    assert result[1].y == pytest.approx(1.5)
    assert result[2].x == pytest.approx(-2.0)
    assert result[2].y == pytest.approx(3.46410, rel=0.01)
    assert result[3].x == pytest.approx(0.59808, rel=0.01)
    assert result[3].y == pytest.approx(4.96410, rel=0.01)


def test_get_xy_from_ij():
    surf = xtgeo.RegularSurface(
        xori=0, yori=0, xinc=1, yinc=1, ncol=6, nrow=5, rotation=30
    )

    regsurf_cpp = _internal.regsurf.RegularSurface(surf)

    yflip = 1
    point = regsurf_cpp.get_xy_from_ij(2, 4, yflip)

    print(point.x, point.y)
    assert point.x == pytest.approx(-0.2679491924)
    assert point.y == pytest.approx(4.4641016151)


@pytest.fixture(scope="module", name="get_drogondata")
def fixture_get_drogondata(testdata_path):
    grid = xtgeo.grid_from_file(f"{testdata_path}/3dgrids/drogon/2/geogrid.roff")
    poro = xtgeo.gridproperty_from_file(
        f"{testdata_path}/3dgrids/drogon/2/geogrid--phit.roff"
    )
    facies = xtgeo.gridproperty_from_file(
        f"{testdata_path}/3dgrids/drogon/2/geogrid--facies.roff"
    )

    surf = xtgeo.surface_from_file(
        f"{testdata_path}/surfaces/drogon/1/01_topvolantis.gri"
    )
    return grid, poro, facies, surf


@functimer(output="info")
def test_sample_grid3d_layer(get_drogondata):
    """Test alternative using shadow classes (JRIV)"""
    grid, poro, facies, surf = get_drogondata

    logger.info("Sample the grid...")

    regsurf_cpp = _internal.regsurf.RegularSurface(surf)

    iindex, jindex, depth_top, depth_bot, inactive = regsurf_cpp.sample_grid3d_layer(
        _internal.grid3d.Grid(grid),
        8,  # 8 is the depth index
        2,
        -1,  # number of threads for OpenMP; -1 means let the system decide
    )
    logger.info("Sample the grid... DONE")

    mask = iindex == -1
    iindex = np.where(iindex == -1, 0, iindex)
    jindex = np.where(jindex == -1, 0, jindex)

    depthmap = surf.copy()
    depthmap.values = depth_top

    poromap = surf.copy()
    # for each map node, I want the poro value given ii and jj
    poromap.values = poro.values[iindex, jindex, 0]
    poromap.values.mask = mask

    facimap = surf.copy()
    # for each map node, I want the poro value given ii and jj
    facimap.values = facies.values[iindex, jindex, 0]
    facimap.values.mask = mask

    assert np.allclose(poromap.values.mean(), 0.1974, atol=0.01)


@pytest.fixture(scope="module")
def keep_top_store():
    """To remember the top layer for the single-threaded case."""
    return {"keep_top": None}


@pytest.mark.xfail(reason="Flaky, fails in some cases for unknown reasons")
@pytest.mark.parametrize("num_threads", [1, 2, 4, 8, 16])
def test_sample_grid3d_layer_num_threads(
    get_drogondata, benchmark, num_threads, keep_top_store
):
    """Benchmark the sampling of the grid for different number of threads."""
    grid, _, _, surf = get_drogondata
    available_threads = os.cpu_count()

    def sample_grid():
        regsurf_cpp = _internal.regsurf.RegularSurface(surf)
        return regsurf_cpp.sample_grid3d_layer(
            _internal.grid3d.Grid(grid),
            8,  # 8 is the depth index
            2,
            num_threads,
        )

    _, _, top, bot, inactive = benchmark(sample_grid)

    top[np.isnan(top)] = 0
    if num_threads == 1:
        keep_top_store["keep_top"] = np.copy(top)  # Ensure a deep copy is stored

    # check if the top layer is exactly the same for all threads; the test is
    # somehwat flaky for many threads (race condition?), so we only check for < half
    # of the threads available.
    if num_threads < available_threads / 2:
        assert np.array_equal(top, keep_top_store["keep_top"]), (
            f"Mismatch with {num_threads} threads"
        )


@pytest.mark.parametrize(
    "x, y, rotation, expected",
    [
        (2.746, 1.262, 0, 17.73800),
        (2.096, 4.897, 0, 17.473),
        (2.746, 2.262, 0, np.nan),
        (1.9999999999, 1.999999999, 0, 14.0),
        (1999, 1999, 0, np.nan),  # far outside
        (2.746, 1.262, 10, 18.3065),
        (2.096, 4.897, 10, 21.9457),
        (2.746, 1.262, 20, 18.319),
        (2.096, 4.897, 20, 25.751),
        (-0.470, -3.354, 210, np.nan),
        (-0.013, -5.148, 210, 19.963),  # rms gets 19.959
        (-1.433, -5.359, 210, 27.448),  # rms gets 27.452
    ],
)
def test_get_z_from_xy_simple(x, y, rotation, expected):
    """Test values are manually verified by inspection in RMS"""
    values = np.array(range(30)).reshape(5, 6).astype(float)

    values[2, 3] = np.nan
    values[4, 5] = np.nan

    surf = xtgeo.RegularSurface(
        xori=0, yori=0, xinc=1, yinc=1, ncol=5, nrow=6, rotation=rotation, values=values
    )

    regsurf_cpp = _internal.regsurf.RegularSurface(surf)
    z_value = regsurf_cpp.get_z_from_xy(x, y)

    if np.isnan(expected):
        assert np.isnan(z_value)
    else:
        assert z_value == pytest.approx(expected, abs=0.001)


@functimer(output="info")
def test_get_z_from_xy(get_drogondata):
    _, _, _, srf = get_drogondata

    surf = srf.copy()

    regsurf_cpp = _internal.regsurf.RegularSurface(surf)
    z_value = regsurf_cpp.get_z_from_xy(460103.00, 5934855.00)

    assert z_value == pytest.approx(1594.303, abs=0.001)

    # make ntimes random points and check the time (cf. @functimer with debug logging)
    ntimes = 100000
    xmin = 458000
    xmax = 468000
    ymin = 5927000
    ymax = 5938000
    logger.debug("Start random points loop... %s", ntimes)

    for _ in range(ntimes):
        x = np.random.uniform(xmin, xmax)
        y = np.random.uniform(ymin, ymax)
        regsurf_cpp.get_z_from_xy(x, y)
    logger.debug("End random points loop")
