"""Test some basic _internal functions which are in C++ and use the pybind11 method.

Some basic methods are tested here, while the more complex ones are tested in an
integrated manner in the other more general tests (like testing surfaces, cubes,
3D grids)

This module focus on testing the C++ functions that are used in the "grid3d"
name space.

"""

import numpy as np
import pytest

import xtgeo
import xtgeo._internal as _internal  # type: ignore
from xtgeo.common.log import null_logger

logger = null_logger(__name__)


@pytest.mark.parametrize(
    "cell, expected_firstcorner, expected_lastcorner",
    [
        (
            (0, 0, 0),
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
        ),
        (
            (1, 0, 0),
            (1.0, 0.0, 0.0),
            (2.0, 1.0, 1.0),
        ),
        (
            (2, 3, 4),
            (2.0, 3.0, 4.0),
            (3.0, 4.0, 5.0),
        ),
    ],
)
def test_cell_corners(cell, expected_firstcorner, expected_lastcorner):
    """Test the cell_corners function, which returns a list from C++."""
    # Create a simple grid
    grid = xtgeo.create_box_grid((3, 4, 5))

    # Get the corners of the first cell
    corners = _internal.grid3d.cell_corners(
        *cell, grid.ncol, grid.nrow, grid.nlay, grid._coordsv, grid._zcornsv
    )

    assert isinstance(corners, list)
    corners = np.array(corners).reshape((8, 3))
    assert np.allclose(corners[0], expected_firstcorner)
    assert np.allclose(corners[7], expected_lastcorner)


def test_cell_corners_minmax(testdata_path):
    """Test the cell_minmax function, which returns a list from C++."""
    # Read the banal6 grid

    cell = (0, 0, 0)
    grid = xtgeo.grid_from_file(f"{testdata_path}/3dgrids/etc/banal6.roff")
    corners = _internal.grid3d.cell_corners(
        *cell, grid.ncol, grid.nrow, grid.nlay, grid._coordsv, grid._zcornsv
    )
    # Get the min and max of the first cell
    minmax = _internal.grid3d.get_corners_minmax(corners)

    print(minmax)
    assert isinstance(minmax, list)
    assert len(minmax) == 6
    assert np.allclose(minmax, [0.0, 25.0, 0.0, 50.0, -0.5, 1.25])


@pytest.mark.parametrize(
    "x, y, cell, position, expected",
    [
        (0.5, 0.5, (0, 0, 0), "top", True),
        (0.5, 0.5, (0, 0, 0), "base", True),
        (0.01, 0.99, (0, 0, 0), "base", True),
        (0.99, 0.99, (0, 0, 0), "base", True),
        (1.5, 0.5, (0, 0, 0), "top", False),
        (1.5, 0.5, (0, 0, 0), "base", False),
    ],
)
def test_is_xy_point_in_cell(x, y, cell, position, expected):
    """Test the XY point is inside a hexahedron cell, seen from top or base."""

    grid = xtgeo.create_box_grid((3, 4, 5))
    corners = _internal.grid3d.cell_corners(
        *cell, grid.ncol, grid.nrow, grid.nlay, grid._coordsv, grid._zcornsv
    )
    assert (
        _internal.grid3d.is_xy_point_in_cell(
            x,
            y,
            corners,
            0 if position == "top" else 1,
        )
        is expected
    )


@pytest.mark.parametrize(
    "x, y, cell, position, expected",
    [
        (61.89, 38.825, (2, 0, 0), "top", 0.27765),
        (89.65, 90.32, (3, 1, 0), "top", 0.26906),
        (95.31, 65.316, (3, 1, 0), "base", 1.32339),
        (999.31, 999.316, (3, 1, 0), "base", np.nan),
    ],
)
def test_get_depth_in_cell(testdata_path, x, y, cell, position, expected):
    """Test the get_depth_in_cell function, which returns a float from C++."""
    # Read the banal6 grid
    grid = xtgeo.grid_from_file(f"{testdata_path}/3dgrids/etc/banal6.roff")

    corners = _internal.grid3d.cell_corners(
        *cell, grid.ncol, grid.nrow, grid.nlay, grid._coordsv, grid._zcornsv
    )
    # Get the depth of the first cell
    depth = _internal.grid3d.get_depth_in_cell(
        x,
        y,
        corners,
        0 if position == "top" else 1,
    )

    if np.isnan(expected):
        assert np.isnan(depth)
    else:
        assert depth == pytest.approx(expected)
