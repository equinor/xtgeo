import numpy as np
import pytest

import xtgeo
import xtgeo._internal as _internal  # type: ignore
from xtgeo._internal.geometry import PointInHexahedronMethod as M  # type: ignore
from xtgeo._internal.xyz import Point  # type: ignore
from xtgeo.common.log import functimer

METHODS = [
    M.RayCasting,
    M.Tetrahedrons,
    M.UsingPlanes,
    M.Isoparametric,
    M.Legacy,
    M.Optimized,
]


IDS = [
    "RayCasting",
    "Tetrahedrons (by scheme)",
    "UsingPlanes",
    "Isoparametric",
    "Legacy",
    "Optimized",
]


@pytest.fixture
def simple_grid():
    """Create a simple 1x1x1 grid with regular geometry for testing."""
    # Create a simple grid with known coordinates
    ncol, nrow, nlay = 1, 1, 1
    xinc, yinc, zinc = 100.0, 100.0, 10.0
    rotation = 0.0
    xori, yori, zori = 0.0, 0.0, 1000.0
    grd = xtgeo.create_box_grid(
        (ncol, nrow, nlay),
        increment=(xinc, yinc, zinc),
        origin=(xori, yori, zori),
        rotation=rotation,
    )
    return _internal.grid3d.Grid(grd)


@pytest.fixture
def complex_grid():
    """Create a more complex 3x3x3 grid with irregular geometry."""
    ncol, nrow, nlay = 3, 3, 3
    xinc, yinc, zinc = 100.0, 100.0, 50.0
    rotation = 30.0  # rotation to test non-aligned grid
    xori, yori, zori = 0.0, 0.0, 1000.0
    grid = xtgeo.create_box_grid(
        (ncol, nrow, nlay),
        increment=(xinc, yinc, zinc),
        origin=(xori, yori, zori),
        rotation=rotation,
    )
    # Make the grid more irregular by modifying some z values
    zcorn_3d = grid._zcornsv.copy()
    # Modify some z values to make irregular cells
    zcorn_3d[1, 1, 1, :] += 20.0  # Shift middle points up
    grid._zcornsv = zcorn_3d
    return _internal.grid3d.Grid(grid)


@pytest.mark.parametrize("method", METHODS, ids=IDS)
@pytest.mark.parametrize(
    "point, expected",
    [
        # Test center point with different methods
        ((50.0, 50.0, 1005.0), True),
        # Test points near boundaries
        ((1.0, 50.0, 1005.0), True),
        ((99.0, 50.0, 1005.0), True),
        ((50.0, 1.0, 1005.0), True),
        ((50.0, 99.0, 1005.0), True),
        ((50.0, 50.0, 1001.0), True),
        ((50.0, 50.0, 1009.0), True),
        ((0.0001, 0.0001, 1009.999), True),  # Point very close to the edge
        # Test points outside
        ((-0.0001, -0.0001, 1010.001), False),  # Point slightly outside
        ((-10.0, 50.0, 1005.0), False),
        ((110.0, 50.0, 1005.0), False),
        ((50.0, -10.0, 1005.0), False),
        ((50.0, 110.0, 1005.0), False),
        ((50.0, 50.0, 990.0), False),
        ((50.0, 50.0, 1020.0), False),
    ],
)
def test_point_inside_cell_methods(simple_grid, method, point, expected):
    """Test different methods for point-in-hexahedron with various test points."""
    cell_corners = simple_grid.get_cell_corners_from_ijk(0, 0, 0)
    point = Point(*point)

    result = _internal.grid3d.is_point_in_cell(point, cell_corners, method)
    assert result == expected, (
        f"Method {method} with point {point} returned {result}, expected {expected}"
    )


def test_point_inside_hexahedroncell_etc_speed(simple_grid):
    """Compare the speed of helper functions for point-in-hexahedron."""
    cell_corners = simple_grid.get_cell_corners_from_ijk(0, 0, 0)

    iterations = 100000

    @functimer(output="print", comment=f"Using {iterations} iterations")
    def convexity_test():
        for i in range(iterations):
            _internal.grid3d.is_cell_non_convex(cell_corners)

    @functimer(output="print", comment=f"Using {iterations} iterations")
    def distorted_test():
        for i in range(iterations):
            _internal.grid3d.is_cell_distorted(cell_corners)

    convexity_test()
    distorted_test()


@pytest.mark.parametrize("method", METHODS, ids=IDS)
def test_point_inside_hexahedron_speed(simple_grid, method):
    """Benchmark the speed of different methods for point-in-hexahedron."""
    cell_corners = simple_grid.get_cell_corners_from_ijk(0, 0, 0)
    center = Point(50.0, 50.0, 1005.0)
    iterations = 100000

    @functimer(output="print", comment=f"Using {iterations} iterations")
    def benchmark_method():
        for _ in range(iterations):
            _internal.grid3d.is_point_in_cell(center, cell_corners, method)

    benchmark_method()


@pytest.mark.parametrize("method", METHODS, ids=IDS)
@pytest.mark.parametrize(
    "point, expected_result, description",
    [
        # Inside points
        (Point(50.0, 50.0, 1005.0), True, "Center point"),
        (Point(40.0, 60.0, 1003.0), True, "Point slightly off center"),
        (Point(0.0, 50.0, 1005.0), True, "Point exactly on the edge"),
        # Outside points
        (Point(-10.0, 50.0, 1005.0), False, "Point outside X bounds"),
        (Point(50.0, 150.0, 1005.0), False, "Point outside Y bounds"),
        (Point(50.0, 50.0, 950.0), False, "Point outside Z bounds"),
    ],
)
def test_point_inside_simple_grid_cell(
    method, simple_grid, point, expected_result, description
):
    """Test if points are correctly identified as inside/outside a simple grid cell."""
    # Get the first cell of the grid
    cell_corners = simple_grid.get_cell_corners_from_ijk(0, 0, 0)

    result = _internal.grid3d.is_point_in_cell(point, cell_corners, method)
    assert result == expected_result, (
        f"Failed for {description}: {point}, got {result}, expected {expected_result}"
    )


@pytest.mark.parametrize("method", METHODS, ids=IDS)
def test_point_inside_complex_grid(complex_grid, method):
    """Test points inside/outside cells of a more complex grid."""
    # Test points in various cells of the grid
    for i in range(3):
        for j in range(3):
            for k in range(3):
                cell_corners = complex_grid.get_cell_corners_from_ijk(i, j, k)

                # Get cell center by averaging corners
                corners_array = np.array(cell_corners.to_numpy()).reshape(-1, 3)
                center_x = np.mean(corners_array[:, 0])
                center_y = np.mean(corners_array[:, 1])
                center_z = np.mean(corners_array[:, 2])

                # Point at center should be inside
                center_point = Point(center_x, center_y, center_z)
                assert _internal.grid3d.is_point_in_cell(
                    center_point, cell_corners, method
                )

                # Point far outside should be outside
                far_point = Point(center_x + 1000, center_y + 1000, center_z + 1000)
                assert not _internal.grid3d.is_point_in_cell(
                    far_point, cell_corners, method
                )


@pytest.mark.parametrize("method", METHODS, ids=IDS)
def test_edge_cases_for_methods(simple_grid, method):
    """Test edge cases for all methods."""
    cell_corners = simple_grid.get_cell_corners_from_ijk(0, 0, 0)

    # Points exactly on the boundary
    on_top_face = Point(50.0, 50.0, 1000.0)
    on_vertex = Point(0.0, 0.0, 1000.0)
    on_edge = Point(50.0, 0.0, 1000.0)

    assert _internal.grid3d.is_point_in_cell(on_top_face, cell_corners, method)
    assert _internal.grid3d.is_point_in_cell(on_vertex, cell_corners, method)
    assert _internal.grid3d.is_point_in_cell(on_edge, cell_corners, method)


@pytest.mark.parametrize(
    "method, expected_result, _",
    [
        (M.RayCasting, (False, False), "Ray casting known struggle with thin cells"),
        (M.Isoparametric, (True, False), "Known to be true/false for isoparametric"),
        (M.UsingPlanes, (True, False), "Known to be true/false for using_planes"),
        (M.Legacy, (True, False), "Known to be true/false for legacy"),
    ],
)
def test_point_inside_thin_cell(method, expected_result, _):
    """Test with a degenerate cell (nearly flat in one direction)."""
    # Create a custom cell that's very thin in z-direction
    p1 = Point(0.0, 0.0, 1000.0)
    p2 = Point(100.0, 0.0, 1000.0)
    p3 = Point(0.0, 100.0, 1000.0)
    p4 = Point(100.0, 100.0, 1000.0)

    p5 = Point(0.0, 0.0, 1000.1)
    p6 = Point(100.0, 0.0, 1000.1)
    p7 = Point(0.0, 100.0, 1000.1)
    p8 = Point(100.0, 100.0, 1000.1)

    cell_corners = _internal.grid3d.CellCorners(p1, p2, p4, p3, p5, p6, p8, p7)

    expected1, expected2 = expected_result

    # Test with point inside the thin cell
    inside_point = Point(50.0, 50.0, 1000.05)
    assert (
        _internal.grid3d.is_point_in_cell(inside_point, cell_corners, method)
        is expected1
    )

    # Test with point just outside the z bounds
    outside_point = Point(50.0, 50.0, 1001.1)
    assert (
        _internal.grid3d.is_point_in_cell(outside_point, cell_corners, method)
        is expected2
    )


@pytest.mark.parametrize(
    "method, expected_result, _",
    [
        (M.RayCasting, (False, False), "Ray casting struggle with deformed cells"),
        (M.UsingPlanes, (True, False), "Known to be true/false for using_planes"),
        (M.Isoparametric, (True, False), "Known to be true/false for isoparametric"),
        (M.Legacy, (True, False), "Known to be true/false for legacy"),
    ],
)
def test_point_inside_deformed_case1_cell(method, expected_result, _):
    """Test with a degenerate cell (deformed, case 1)."""
    # Create a custom cell that's very thin in z-direction
    p1 = Point(0.0, 0.0, 1000.0)
    p2 = Point(10.0, 0.0, 1000.0)
    p3 = Point(0.0, 200.0, 1000.0)
    p4 = Point(100.0, 100.0, 1000.0)

    p5 = Point(0.0, 0.0, 1010.0)
    p6 = Point(10.0, 0.0, 1010.0)
    p7 = Point(0.0, 200.0, 1010.0)
    p8 = Point(100.0, 100.0, 1010.0)

    cell_corners = _internal.grid3d.CellCorners(p1, p2, p4, p3, p5, p6, p8, p7)

    expected1, expected2 = expected_result

    # Test with point inside deformed cell
    inside_point = Point(50.0, 50.0, 1005.0)
    assert (
        _internal.grid3d.is_point_in_cell(inside_point, cell_corners, method)
        is expected1
    )

    # Test with point just outside the bounds
    outside_point = Point(60.0, 50.0, 1005.0)
    assert (
        _internal.grid3d.is_point_in_cell(outside_point, cell_corners, method)
        is expected2
    )


@pytest.mark.parametrize(
    "method, inside_point, outside_point, expected_inside, expected_outside",
    [
        (
            M.Isoparametric,
            Point(50.0, 50.0, 1000.5),
            Point(60.0, 50.0, 1000.5),
            True,
            False,
        ),
        (
            M.UsingPlanes,
            Point(50.0, 50.0, 1000.5),
            Point(60.0, 50.0, 1000.5),
            True,
            False,
        ),
        (
            M.RayCasting,
            Point(50.0, 50.0, 1000.5),
            Point(60.0, 50.0, 1000.5),
            False,
            False,  # expected to be True
        ),
        (
            M.Legacy,
            Point(50.0, 50.0, 1000.5),
            Point(60.0, 50.0, 1000.5),
            True,
            False,
        ),
    ],
)
def test_point_inside_deformed_case2_cell_with_parametrize(
    method, inside_point, outside_point, expected_inside, expected_outside
):
    """Test with a degenerate cell (deformed, case 2) using parameterized methods."""
    # Create a custom deformed cell that's very thin in z-direction
    p1 = Point(0.0, 0.0, 1000.0)
    p2 = Point(100.0, 0.0, 1000.0)
    p3 = Point(0.0, 100.0, 1000.0)
    p4 = Point(10.0, 10.0, 1000.0)

    p5 = Point(0.0, 0.0, 1001.0)
    p6 = Point(100.0, 0.0, 1001.0)
    p7 = Point(0.0, 100.0, 1001.0)
    p8 = Point(10.0, 10.0, 1001.0)

    cell_corners = _internal.grid3d.CellCorners(p1, p2, p4, p3, p5, p6, p8, p7)

    # Check if the cell is non-convex
    assert _internal.grid3d.is_cell_non_convex(cell_corners) is True

    # Test with the inside point
    assert (
        _internal.grid3d.is_point_in_cell(inside_point, cell_corners, method)
        == expected_inside
    )

    # Test with the outside point
    assert (
        _internal.grid3d.is_point_in_cell(outside_point, cell_corners, method)
        == expected_outside
    )


def test_more_vertices():
    vrt = [0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1]
    cell_corners = _internal.grid3d.CellCorners(vrt)
    assert _internal.grid3d.is_cell_non_convex(cell_corners) is False

    assert _internal.grid3d.is_point_in_cell(Point(0.5, 0.5, 0.5), cell_corners) is True
    assert (
        _internal.grid3d.is_point_in_cell(Point(-0.1, 0.5, 0.5), cell_corners) is False
    )


@pytest.fixture
def non_convex_cell():
    """Fixture for a non-convex cell based on a real case."""
    vrt = [
        [461351.493253, 5938298.477428, 1850],
        [461501.758690, 5938385.850231, 1850],
        [461440.718409, 5938166.753852, 1850.1],
        [461582.200838, 5938248.702782, 1850],
        [461354.611430, 5938300.454809, 1883.246948],
        [461504.611754, 5938387.700867, 1915.005005],
        [461443.842986, 5938169.007646, 1904.730957],
        [461585.338388, 5938250.905010, 1921.021973],
    ]
    return _internal.grid3d.CellCorners(np.array(vrt).flatten())


@pytest.mark.parametrize(
    "method, expected_inside, expected_outside",
    [
        (
            M.Isoparametric,
            False,  # expected to be True; hence isoparametric struggles
            False,
        ),
        (
            M.UsingPlanes,
            True,
            False,
        ),
        (
            M.RayCasting,
            True,
            False,
        ),
        (
            M.Legacy,
            True,
            False,
        ),
    ],
)
def test_non_convex_cell_methods(
    non_convex_cell, method, expected_inside, expected_outside
):
    """Test a non-convex cell with various methods."""
    # Points to test
    inside_point = Point(461467.513586, 5938273.910537, 1850)
    outside_point = Point(461467.513586, 5938273.910537, 1849.95)

    # Check if the cell is non-convex
    assert _internal.grid3d.is_cell_non_convex(non_convex_cell) is True, (
        "Expected the cell to be non-convex, but it was not detected as such."
    )

    # Test the inside point
    result_inside = _internal.grid3d.is_point_in_cell(
        inside_point, non_convex_cell, method
    )
    assert result_inside is expected_inside, (
        f"Method {method} failed for inside point {inside_point}. "
        f"Expected True, got {result_inside}."
    )

    # Test the outside point
    result_outside = _internal.grid3d.is_point_in_cell(
        outside_point, non_convex_cell, method
    )
    assert result_outside is expected_outside, (
        f"Method {method} failed for outside point {outside_point}. "
        f"Expected False, got {result_outside}."
    )


def test_tricky_drogon_cell():
    """Test a tricky cell where ambiguity exists, being between two cells."""
    cell_110_171_81 = [
        [465036.45, 5931366.81, 1673.49710],
        [465077.85, 5931386.67, 1678.6278],
        [465061.39, 5931322.43, 1675.6483],
        [465102.27, 5931341.11, 1681.5220],
        [465035.76, 5931364.89, 1674.6874],
        [465077.14, 5931384.61, 1679.8429],
        [465060.68, 5931320.56, 1676.8409],
        [465101.54, 5931339.11, 1682.7357],
    ]
    cell_110_172_81 = [
        [465061.39, 5931322.43, 1675.6483],
        [465102.27, 5931341.11, 1681.5220],
        [465085.99, 5931277.28, 1678.5707],
        [465126.57, 5931295.41, 1684.7965],
        [465060.68, 5931320.56, 1676.8409],
        [465101.54, 5931339.11, 1682.7357],
        [465085.27, 5931275.47, 1679.7644],
        [465125.83, 5931293.46, 1686.0125],
    ]
    cell_110_171_one = [  # the full column from "onegrid"
        [465048.57, 5931400.53, 1652.5778],
        [465090.32, 5931422.84, 1657.2687],
        [465073.71, 5931354.95, 1654.8762],
        [465114.99, 5931376.05, 1660.3265],
        [465022.65, 5931328.44, 1697.3019],
        [465063.65, 5931345.52, 1702.9298],
        [465047.24, 5931285.09, 1699.5015],
        [465087.71, 5931301.10, 1705.7964],
    ]
    cr1 = _internal.grid3d.CellCorners(np.array(cell_110_171_81).flatten())
    cr2 = _internal.grid3d.CellCorners(np.array(cell_110_172_81).flatten())
    cr3 = _internal.grid3d.CellCorners(np.array(cell_110_171_one).flatten())

    p = Point(465100.000, 5931340.000, 1681.288)

    assert _internal.grid3d.is_point_in_cell(p, cr1, M.Isoparametric) is True
    assert _internal.grid3d.is_point_in_cell(p, cr2, M.Isoparametric) is False
    assert _internal.grid3d.is_point_in_cell(p, cr3, M.Isoparametric) is True

    # this shows that RayCasting is not the best method for this case
    assert _internal.grid3d.is_point_in_cell(p, cr1, M.RayCasting) is True
    assert _internal.grid3d.is_point_in_cell(p, cr2, M.RayCasting) is False
    assert _internal.grid3d.is_point_in_cell(p, cr3, M.RayCasting) is False

    assert _internal.grid3d.is_point_in_cell(p, cr1, M.Tetrahedrons) is True
    assert _internal.grid3d.is_point_in_cell(p, cr2, M.Tetrahedrons) is False
    assert _internal.grid3d.is_point_in_cell(p, cr3, M.Tetrahedrons) is True

    assert _internal.grid3d.is_point_in_cell(p, cr1, M.Legacy) is True
    assert _internal.grid3d.is_point_in_cell(p, cr2, M.Legacy) is False
    assert _internal.grid3d.is_point_in_cell(p, cr3, M.Legacy) is True

    # this shows that UsingPlanes is not the best method for this case
    assert _internal.grid3d.is_point_in_cell(p, cr1, M.UsingPlanes) is True
    assert _internal.grid3d.is_point_in_cell(p, cr2, M.UsingPlanes) is False
    assert _internal.grid3d.is_point_in_cell(p, cr3, M.UsingPlanes) is False
