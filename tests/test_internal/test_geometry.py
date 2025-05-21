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
    corners = _internal.grid3d.Grid(grid).get_cell_corners_from_ijk(0, 0, 0)

    for prec in [
        _internal.geometry.HexVolumePrecision.P1,
        _internal.geometry.HexVolumePrecision.P2,
        _internal.geometry.HexVolumePrecision.P4,
    ]:
        volume = _internal.geometry.hexahedron_volume(corners, prec)
        assert volume == pytest.approx(1.0)


def test_simple_cell_volume():
    """Test volumes using the overload cellcorners"""

    vertices = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ]

    known_volume = 1.0

    vert = np.array(vertices, dtype=np.float64).flatten()
    cell = _internal.grid3d.CellCorners(vert)
    # Get the volume
    volume_new = _internal.geometry.hexahedron_volume(cell, _internal.geometry.P2)
    assert volume_new == pytest.approx(known_volume)


def test_distorted_v1_cell_volume():
    """Distorted cell with known volume"""

    vertices = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 0.5, 1.0],
        [1.0, 1.0, 1.0],
    ]

    known_volume = 0.75

    vert = np.array(vertices, dtype=np.float64).flatten()
    cell = _internal.grid3d.CellCorners(vert)
    # Get the volume
    volume = _internal.geometry.hexahedron_volume(
        cell, _internal.geometry.HexVolumePrecision.P2
    )
    assert volume == pytest.approx(known_volume)


def test_distorted_v2_cell_volume():
    """Distorted cell with known volume, here the cells is a triangle"""

    vertices = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ]

    known_volume = 0.5

    vert = np.array(vertices, dtype=np.float64).flatten()
    cell = _internal.grid3d.CellCorners(vert)
    # Get the volume
    volume = _internal.geometry.hexahedron_volume(
        cell, _internal.geometry.HexVolumePrecision.P2
    )
    assert volume == pytest.approx(known_volume)


def test_distorted_v2_rev_cell_volume():
    """As v2 but cells are mirrored"""

    vertices = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ]

    known_volume = 0.5

    vert = np.array(vertices, dtype=np.float64).flatten()
    cell = _internal.grid3d.CellCorners(vert)
    # Get the volume
    volume = _internal.geometry.hexahedron_volume(
        cell, _internal.geometry.HexVolumePrecision.P2
    )
    assert volume == pytest.approx(known_volume)


def test_distorted_v3_cell_volume():
    """Distorted cell with known volume, here the cells is just collapsed"""

    vertices = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ]

    known_volume = 0.0

    vert = np.array(vertices, dtype=np.float64).flatten()
    cell = _internal.grid3d.CellCorners(vert)
    # Get the volume
    volume_new = _internal.geometry.hexahedron_volume(
        cell, _internal.geometry.HexVolumePrecision.P2
    )
    assert volume_new == pytest.approx(known_volume)


def test_distorted_v4_cell_volume():
    """Distorted cell with known volume, here the cells are almost collapsed"""

    vertices = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.99, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.99, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ]

    known_volume = 0.005

    vert = np.array(vertices, dtype=np.float64).flatten()
    cell = _internal.grid3d.CellCorners(vert)
    # Get the volume
    volume_new = _internal.geometry.hexahedron_volume(
        cell, _internal.geometry.HexVolumePrecision.P2
    )
    assert volume_new == pytest.approx(known_volume)


def test_cell_volume_speed():
    """Compare the speed of cell colume calculations"""

    vertices = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ]

    vert = np.array(vertices, dtype=np.float64).flatten()
    cell_corners = _internal.grid3d.CellCorners(vert)

    iterations = 100000
    comment = f"Using {iterations} iterations"

    @functimer(output="print", comment=comment)
    def volume_2():
        for i in range(iterations):
            _internal.geometry.hexahedron_volume(
                cell_corners, _internal.geometry.HexVolumePrecision.P2
            )

    @functimer(output="print", comment=comment)
    def volume_4():
        for i in range(iterations):
            _internal.geometry.hexahedron_volume(
                cell_corners, _internal.geometry.HexVolumePrecision.P4
            )

    volume_2()
    volume_4()


def test_hexahedron_volume_my_banal6_cell(testdata_path):
    """Test the hexahedron function using banal6 synth case"""
    # Read the banal6 grid
    grid = xtgeo.grid_from_file(f"{testdata_path}/3dgrids/etc/banal6.roff")

    # Get the corners of a skew cell (2,1,2 in RMS using 1-based indexing)
    corners = _internal.grid3d.Grid(grid).get_cell_corners_from_ijk(1, 0, 1)

    precrange = [
        _internal.geometry.HexVolumePrecision.P1,
        _internal.geometry.HexVolumePrecision.P2,
        _internal.geometry.HexVolumePrecision.P4,
    ]

    for prec in precrange:
        volume1 = _internal.geometry.hexahedron_volume(corners, prec)
        assert volume1 == pytest.approx(1093.75, rel=1e-3)  # 1093.75 is RMS' value

    # Get the corners of a another skew cell (4,1,2)
    corners = _internal.grid3d.Grid(grid).get_cell_corners_from_ijk(3, 0, 1)

    for prec in precrange:
        volume2 = _internal.geometry.hexahedron_volume(corners, prec)
        assert volume2 == pytest.approx(468.75, rel=1e-3)  # 468.75 is RMS' value

    # some work on the corners
    corners_np = corners.to_numpy()
    assert corners_np.shape == (8, 3)
    assert corners_np.dtype == np.float64
    assert corners_np[0, 0] == corners.upper_sw.x
    assert corners_np[7, 2] == corners.lower_ne.z


def test_is_xy_point_in_polygon():
    """Test the point_is_inside_polygon function, which returns a bool from C++."""

    # Create a simple polygon (list or np array)
    polygon = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    polygon = _internal.xyz.Polygon(polygon)

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
    poly = np.stack(
        [df["X_UTME"].values, df["Y_UTMN"].values, df["Z_TVDSS"].values], axis=1
    )  # ~ 100k points
    logger.debug(f"Polygon has {len(poly)} points")

    poly = _internal.xyz.Polygon(poly)

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

    p1 = _internal.xyz.Point(*c1)
    p2 = _internal.xyz.Point(*c2)
    p3 = _internal.xyz.Point(*c3)
    p4 = _internal.xyz.Point(*c4)

    x_point = 93.19
    y_point = 96.00

    z = _internal.geometry.interpolate_z_4p(x_point, y_point, p1, p2, p3, p4)
    assert z == pytest.approx(0.37818, rel=1e-3)

    x_point = 999.00
    y_point = 96.00

    z = _internal.geometry.interpolate_z_4p(x_point, y_point, p1, p2, p3, p4)
    assert np.isnan(z)


def test_z_value_in_regular_grid():
    """Test the interpolate_z_4p_regular function, which returns a float from C++."""

    points = [
        _internal.xyz.Point(75.0, 50.0, 0.0),
        _internal.xyz.Point(100.0, 50.0, 0.0),
        _internal.xyz.Point(75.0, 100.0, 0.0),
        _internal.xyz.Point(100.0, 100.0, 0.55),
    ]

    x_point, y_point = 93.19, 96.00
    z = _internal.geometry.interpolate_z_4p_regular(x_point, y_point, *points)
    assert z == pytest.approx(0.36817, rel=1e-3)

    x_point, y_point = 999.00, 96.00
    z = _internal.geometry.interpolate_z_4p_regular(x_point, y_point, *points)
    assert np.isnan(z)


def test_xy_point_in_quadrilateral():
    """Test the is_xy_point_in_quadrilateral function, which returns a bool from C++."""

    points = [
        _internal.xyz.Point(75.0, 50.0, 0.0),
        _internal.xyz.Point(100.0, 50.0, 0.0),
        _internal.xyz.Point(100.0, 100.0, 0.55),
        _internal.xyz.Point(75.0, 100.0, 0.0),
    ]

    assert _internal.geometry.is_xy_point_in_quadrilateral(90, 75, *points)
    assert _internal.geometry.is_xy_point_in_quadrilateral(90.0, 75.0, *points)
    assert _internal.geometry.is_xy_point_in_quadrilateral(75.001, 50.001, *points)
    assert not _internal.geometry.is_xy_point_in_quadrilateral(0, 0, *points)


@pytest.mark.parametrize(
    "point, tolerance, expected",
    [
        ((75, 50), 1e-19, True),
        ((75, 50), 1e-2, True),
        ((74.9999999, 50), 1e-19, False),
        ((74.9999999, 50), 1e-2, True),  # is actually outside, but tolerance is low
        ((75, 50.0000001), 1e-19, True),
        ((75, 50.0000001), 1e-2, True),
        ((75.0000001, 50.0000001), 1e-19, True),
        ((75.0000001, 50.0000001), 1e-2, True),
    ],
)
def test_xy_point_in_quadrilateral_tolerance(point, tolerance, expected):
    """Test the is_xy_point_in_quadrilateral function, tolerance parameter."""

    points = [
        _internal.xyz.Point(75.0, 50.0, 0.0),
        _internal.xyz.Point(100.0, 50.0, 0.0),
        _internal.xyz.Point(100.0, 100.0, 0.55),
        _internal.xyz.Point(75.0, 100.0, 0.0),
    ]

    assert (
        _internal.geometry.is_xy_point_in_quadrilateral(*point, *points, tolerance)
        is expected
    )


@pytest.mark.parametrize(
    "point, tolerance, expected",
    [
        ((0.09, 0.09), 1e-19, True),
        ((0.1, 0.1), 1e-19, True),
        ((0.9999999, 0.9999999), 1e-19, True),
        ((1.0000001, 1.0000001), 1e-19, False),
    ],
)
def test_xy_point_in_quadrilateral_skew(point, tolerance, expected):
    """Test the is_xy_point_in_quadrilateral function, unordered corners.

    The quadrilateral is rather scewed, so the point appears not inside the
    quadrilateral, but that is wrong. Meaning that the function is only safe for
    "convex" quadrilaterals.
    """

    c1 = [0, 0, 0]
    c2 = [4, 0, 0]
    c3 = [1, 1, 0]
    c4 = [0, 4, 0]

    p1 = _internal.xyz.Point(*c1)
    p2 = _internal.xyz.Point(*c2)
    p3 = _internal.xyz.Point(*c3)
    p4 = _internal.xyz.Point(*c4)

    assert (
        _internal.geometry.is_xy_point_in_quadrilateral(
            *point, p1, p2, p3, p4, tolerance
        )
        is expected
    )
    assert (
        _internal.geometry.is_xy_point_in_quadrilateral(
            *point, p1, p4, p3, p2, tolerance
        )
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

    p1 = _internal.xyz.Point(*c1)
    p2 = _internal.xyz.Point(*c2)
    p3 = _internal.xyz.Point(*c3)
    p4 = _internal.xyz.Point(*c4)

    def run_benchmark():
        _internal.geometry.is_xy_point_in_quadrilateral(*point, p1, p2, p3, p4)

    benchmark(run_benchmark)


def test_xy_point_in_polygon_speed(benchmark):
    """Benchmark the is_xy_point_in_polygon function"""

    # Create a simple quadrilateral (list or np array)
    c1 = [75.0, 50.0, 0.0]
    c2 = [100.0, 50.0, 0.0]
    c3 = [100.0, 100.0, 0.55]
    c4 = [75.0, 100.0, 0.0]

    poly = _internal.xyz.Polygon(np.array([c1, c2, c3, c4]))
    point = (90, 75)

    def run_benchmark():
        _internal.geometry.is_xy_point_in_polygon(*point, poly)

    benchmark(run_benchmark)
