"""Test the grid3d points inside function.

Here focus is on a Points array and the full 3D grid, not each cells
(cf test_grid3d_cell_point_inside.py).

"""

import numpy as np
import pytest
import xtgeo._internal as _internal  # type: ignore
from xtgeo._internal.geometry import PointInHexahedronMethod as M  # type: ignore

import xtgeo
from xtgeo.common.log import functimer


@pytest.fixture
def small_grid():
    """Create a 3x3x3 grid with simple geometry."""
    ncol, nrow, nlay = 3, 3, 3
    xinc, yinc, zinc = 100.0, 100.0, 50.0
    rotation = 0.0  # rotation to test non-aligned grid
    xori, yori, zori = 0.0, 0.0, 1000.0
    grid = xtgeo.create_box_grid(
        (ncol, nrow, nlay),
        increment=(xinc, yinc, zinc),
        origin=(xori, yori, zori),
        rotation=rotation,
    )
    return grid, _internal.grid3d.Grid(grid)


@pytest.fixture
def large_grid():
    """Create a 10 million cell grid with simple geometry."""
    ncol, nrow, nlay = 100, 1000, 100
    xinc, yinc, zinc = 100.0, 100.0, 1.0
    rotation = 0.0  # rotation to test non-aligned grid
    xori, yori, zori = 0.0, 0.0, 1000.0
    grid = xtgeo.create_box_grid(
        (ncol, nrow, nlay),
        increment=(xinc, yinc, zinc),
        origin=(xori, yori, zori),
        rotation=rotation,
    )
    return grid, _internal.grid3d.Grid(grid)


@pytest.fixture(scope="module", name="drogon_grid")
def fixture_get_drogondata(testdata_path):
    grid = xtgeo.grid_from_file(f"{testdata_path}/3dgrids/drogon/2/geogrid.roff")

    return (grid, _internal.grid3d.Grid(grid))


def test_points_inside_small_grid(small_grid):
    """Test the points_inside function with a small grid."""
    # Create a set of points to test
    points_input = [
        (-10, 2, 1010.0),  # outside the grid
        (2, 2, 1010.0),  # inside the grid in first cell (0, 0, 0)
        (102, 105, 1010.0),  # inside the grid in second cell in IJ (1, 1, 0)
        (299, 299, 1149.0),  # inside the grid in last cell (2, 2, 2)
        (2, 2, 1200.0),  # outside the grid in Z
    ]
    points = xtgeo.Points(points_input)

    arr = points.get_xyz_arrays()
    assert arr is not None

    grid, grid_cpp = small_grid
    cache = grid._get_cache()

    points_cpp = _internal.xyz.PointSet(arr)

    iarr, jarr, karr = grid_cpp.get_indices_from_pointset(
        points_cpp,
        cache.onegrid_cpp,
        cache.top_i_index_cpp,
        cache.top_j_index_cpp,
        cache.base_i_index_cpp,
        cache.base_j_index_cpp,
        cache.top_depth_cpp,
        cache.base_depth_cpp,
        cache.threshold_magic_1,
        False,  # all cells
        M.Optimized,  # use optimized method
    )

    res = np.array([iarr, jarr, karr]).T

    np.testing.assert_array_equal(
        res,
        [[-1, -1, -1], [0, 0, 0], [1, 1, 0], [2, 2, 2], [-1, -1, -1]],
    )


def test_points_inside_drogon(drogon_grid):
    """Test the points_inside function using_drogon."""
    # Create a set of points to test
    grid, _ = drogon_grid

    x_from_grid, y_from_grid, z_from_grid = grid.get_xyz(asmasked=False)
    x_arr = x_from_grid.values[30, 50, 0:40]
    y_arr = y_from_grid.values[30, 50, 0:40]
    z_arr = z_from_grid.values[30, 50, 0:40]

    grid._actnumsv[30, 50, 2] = 0  # set an inactive cell

    arr = np.array([x_arr, y_arr, z_arr]).T

    points_cpp = _internal.xyz.PointSet(arr)

    @functimer(output="print")
    def preprocess():
        return grid._get_cache()

    cache = preprocess()

    @functimer(output="print")
    def calc():  # inner function to compute effective time on this function
        return grid._get_grid_cpp().get_indices_from_pointset(
            points_cpp,
            cache.onegrid_cpp,
            cache.top_i_index_cpp,
            cache.top_j_index_cpp,
            cache.base_i_index_cpp,
            cache.base_j_index_cpp,
            cache.top_depth_cpp,
            cache.base_depth_cpp,
            cache.threshold_magic_1,
            True,  # active cells only
            M.Optimized,  # use optimized method
        )

    # Call the function to get the result and measure the time taken
    iarr, jarr, karr = calc()

    res = np.array([iarr, jarr, karr]).T

    exp_i = np.ones_like(iarr) * 30
    exp_j = np.ones_like(jarr) * 50
    exp_k = np.arange(0, 40)
    expected = np.array([exp_i, exp_j, exp_k]).T
    expected[2, :] = -1  # inactive cell

    np.testing.assert_array_equal(res, expected)


def test_many_points_inside_large_grid(large_grid):
    """Test many point using a large grid."""
    # Create a set of points to test
    grid, _ = large_grid

    poly = xtgeo.Polygons(
        [
            (100, 100, 1000),
            (9900, 9900, 1100),
        ]
    )

    poly.rescale(0.2)
    assert len(poly.get_dataframe()) == 69297

    arr = poly.get_xyz_arrays()
    assert arr is not None

    points_cpp = _internal.xyz.PointSet(arr)

    @functimer(output="print")
    def preprocess():
        return grid._get_cache()

    cache = preprocess()

    comment = (
        f"Grid has {grid.ntotal / 1e6} million cells, Polygon has {poly.nrow} points"
    )

    @functimer(output="print", comment=comment)
    def calc(threshold=None):
        t = threshold if threshold is not None else cache.threshold_magic_1
        return grid._get_grid_cpp().get_indices_from_pointset(
            points_cpp,
            cache.onegrid_cpp,
            cache.top_i_index_cpp,
            cache.top_j_index_cpp,
            cache.base_i_index_cpp,
            cache.base_j_index_cpp,
            cache.top_depth_cpp,
            cache.base_depth_cpp,
            t,
            False,  # all cells
            M.Optimized,  # use optimized method
        )

    # Call the function to get the result and measure the time taken
    iarr1, jarr1, karr1 = calc()

    iarr2, jarr2, karr2 = calc()  # after caching in cpp

    # using too low threshold should give a bit slower result...
    iarr3, jarr3, karr3 = calc(threshold=0.01)

    np.testing.assert_array_equal(iarr1, iarr2)
    np.testing.assert_array_equal(iarr1, iarr3)
    np.testing.assert_array_equal(jarr1, jarr2)
    np.testing.assert_array_equal(jarr1, jarr3)
    np.testing.assert_array_equal(karr1, karr2)
    np.testing.assert_array_equal(karr1, karr3)
