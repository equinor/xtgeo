"""Test some basic _internal functions which are in C++ and use the pybind11 method.

Some basic methods are tested here, while the more complex ones are tested in an
integrated manner in the other more general tests (like testing surfaces, cubes,
3D grids)

This module focus on testing the C++ functions that are used in the "geometry"
name space.
"""

import numpy as np
import pytest

import xtgeo
import xtgeo._internal as _internal  # type: ignore
from xtgeo.common.log import functimer, null_logger

logger = null_logger(__name__)


def test_hexahedron_volume():
    """Test the hexahedron volume function, which returns a double from C++."""
    # Create a simple grid
    grid = xtgeo.create_box_grid((3, 4, 5))

    # Get the corners of the first cell
    corners = _internal.grid3d.cell_corners(
        0, 0, 0, grid.ncol, grid.nrow, grid.nlay, grid._coordsv, grid._zcornsv
    )

    for prec in range(1, 5):
        volume = _internal.geometry.hexahedron_volume(corners, prec)
        assert volume == pytest.approx(1.0)


def test_hexahedron_volume_banal6_cell(testdata_path):
    """Test the hexahedron function using banal6 synth case"""
    # Read the banal6 grid
    grid = xtgeo.grid_from_file(f"{testdata_path}/3dgrids/etc/banal6.roff")

    # Get the corners of a skew cell (2,1,2 in RMS using 1-based indexing)
    corners = _internal.grid3d.cell_corners(
        1, 0, 1, grid.ncol, grid.nrow, grid.nlay, grid._coordsv, grid._zcornsv
    )

    for prec in range(1, 5):
        volume = _internal.geometry.hexahedron_volume(corners, prec)
        assert volume == pytest.approx(1093.75, rel=1e-3)  # 1093.75 is RMS' value

    # Get the corners of a another skew cell (4,1,2)
    corners = _internal.grid3d.cell_corners(
        3, 0, 1, grid.ncol, grid.nrow, grid.nlay, grid._coordsv, grid._zcornsv
    )

    for prec in range(1, 5):
        volume = _internal.geometry.hexahedron_volume(corners, prec)
        assert volume == pytest.approx(468.75, rel=1e-3)  # 468.75 is RMS' value

    # input corners as numpy array in stead of list is accepted
    corners = np.array(corners)
    volume = _internal.geometry.hexahedron_volume(corners, prec)
    assert volume == pytest.approx(468.75, rel=1e-3)  # 468.75 is RMS' value


def test_is_xy_point_in_polygon():
    """Test the point_is_inside_polygon function, which returns a bool from C++."""

    # Create a simple polygon (list or np array)
    polygon = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ]
    )

    # Test some points
    assert _internal.geometry.is_xy_point_in_polygon(0.5, 0.5, polygon)
    assert _internal.geometry.is_xy_point_in_polygon(0.0, 0.0, polygon)  # border
    assert not _internal.geometry.is_xy_point_in_polygon(1.5, 0.5, polygon)


@functimer
def test_is_xy_point_in_polygon_large(testdata_path):
    """Test the point_is_inside_polygon function, now with a large polygon."""

    polygon = xtgeo.polygons_from_file(f"{testdata_path}/polygons/reek/1/mypoly.pol")

    polygon.rescale(0.2)  # make it much larger!
    df = polygon.get_dataframe()
    df = df[df["POLY_ID"] == 0]
    poly = np.stack([df["X_UTME"].values, df["Y_UTMN"].values], axis=1)  # ~ 100k points
    logger.debug(f"Polygon has {len(poly)} points")
    # Test some points
    logger.debug("Asserting points...")
    assert _internal.geometry.is_xy_point_in_polygon(462192, 5928700, poly)
    assert not _internal.geometry.is_xy_point_in_polygon(1.5, 0.5, poly)
    assert _internal.geometry.is_xy_point_in_polygon(
        460092.64069227, 5931584.0480124, poly
    )
    assert not _internal.geometry.is_xy_point_in_polygon(
        460092.64069227 - 1.0, 5931584.0480124, poly
    )
    logger.debug("Asserting points... done")


def test_z_value_in_irregular_grid():
    """Test the interpolate_z_4p function, which returns a float from C++."""

    c1 = [75.0, 50.0, 0.0]
    c2 = [100.0, 50.0, 0.0]
    c3 = [75.0, 100.0, 0.0]
    c4 = [100.0, 100.0, 0.55]

    x_point = 93.19
    y_point = 96.00

    z = _internal.geometry.interpolate_z_4p(x_point, y_point, c1, c2, c3, c4)
    assert z == pytest.approx(0.37818, rel=1e-3)

    x_point = 999.00
    y_point = 96.00

    z = _internal.geometry.interpolate_z_4p(x_point, y_point, c1, c2, c3, c4)
    assert np.isnan(z)


def test_z_value_in_regular_grid():
    """Test the interpolate_z_4p_regular function, which returns a float from C++.

    3                 4
    x-----------------x
    |                 |
    |                 |  ordering of corners is important
    |                 |
    |                 |
    x-----------------x
    1                 2

    """

    c1 = [75.0, 50.0, 0.0]
    c2 = [100.0, 50.0, 0.0]
    c3 = [75.0, 100.0, 0.0]
    c4 = [100.0, 100.0, 0.55]

    x_point = 93.19
    y_point = 96.00

    # note order of corners is clockwise or counter-clockwise
    z = _internal.geometry.interpolate_z_4p_regular(x_point, y_point, c1, c2, c3, c4)
    assert z == pytest.approx(0.36817, rel=1e-3)

    x_point = 999.00
    y_point = 96.00

    z = _internal.geometry.interpolate_z_4p_regular(x_point, y_point, c1, c2, c3, c4)
    assert np.isnan(z)


def test_xy_point_in_quadrilateral():
    """Test the is_xy_point_in_quadrilateral function, which returns a bool from C++."""

    # Create a simple quadrilateral (list or np array)
    c1 = [75.0, 50.0, 0.0]
    c2 = [100.0, 50.0, 0.0]
    c3 = [100.0, 100.0, 0.55]
    c4 = [75.0, 100.0, 0.0]

    # Test some points
    assert _internal.geometry.is_xy_point_in_quadrilateral(90, 75, c1, c2, c3, c4)
    assert _internal.geometry.is_xy_point_in_quadrilateral(
        75.001, 50.001, c1, c2, c3, c4
    )
    assert not _internal.geometry.is_xy_point_in_quadrilateral(0, 0, c1, c2, c3, c4)


@pytest.mark.parametrize(
    "point, expected",
    [
        ((0.09, 0.09), True),
        ((0.1, 0.1), True),
        ((0.9999999, 0.9999999), True),
        ((1.0000001, 1.0000001), False),
    ],
)
def test_xy_point_in_quadrilateral_skew(point, expected):
    """Test the is_xy_point_in_quadrilateral function, unordered corners.

    The quadrilateral is rather scewed, so the point appears not is not inside the
    quadrilateral, but that is wrong. Meaning that the function is only safe for
    "convex" quadrilaterals.
    """

    c1 = [0, 0, 0]
    c2 = [4, 0, 0]
    c3 = [1, 1, 0]
    c4 = [0, 4, 0]

    assert (
        _internal.geometry.is_xy_point_in_quadrilateral(*point, c1, c2, c3, c4)
        is expected
    )
    assert (
        _internal.geometry.is_xy_point_in_quadrilateral(*point, c1, c4, c3, c2)
        is expected
    )


def test_xy_point_in_quadrilateral_speed(benchmark):
    """Benchmark the is_xy_point_in_quadrilateral function"""

    # Create a simple quadrilateral (list or np array)
    c1 = [75.0, 50.0, 0.0]
    c2 = [100.0, 50.0, 0.0]
    c3 = [100.0, 100.0, 0.55]
    c4 = [75.0, 100.0, 0.0]

    point = (90, 75)

    def run_benchmark():
        _internal.geometry.is_xy_point_in_quadrilateral(*point, c1, c2, c3, c4)

    benchmark(run_benchmark)


def test_xy_point_in_polygon_speed(benchmark):
    """Benchmark the is_xy_point_in_polygon function"""

    # Create a simple quadrilateral (list or np array)
    c1 = [75.0, 50.0, 0.0]
    c2 = [100.0, 50.0, 0.0]
    c3 = [100.0, 100.0, 0.55]
    c4 = [75.0, 100.0, 0.0]
    poly = np.array([c1[0:2], c2[0:2], c3[0:2], c4[0:2]])
    point = (90, 75)

    def run_benchmark():
        _internal.geometry.is_xy_point_in_polygon(*point, poly)

    benchmark(run_benchmark)
