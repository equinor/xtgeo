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
from xtgeo.common.log import functimer, null_logger

logger = null_logger(__name__)


@pytest.fixture(scope="module", name="get_drogondata")
def fixture_get_drogondata(testdata_path):
    grid = xtgeo.grid_from_file(f"{testdata_path}/3dgrids/drogon/2/geogrid.roff")
    poro = xtgeo.gridproperty_from_file(
        f"{testdata_path}/3dgrids/drogon/2/geogrid--phit.roff"
    )
    facies = xtgeo.gridproperty_from_file(
        f"{testdata_path}/3dgrids/drogon/2/geogrid--facies.roff"
    )

    return grid, poro, facies


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
    grid_cpp = _internal.grid3d.Grid(grid)
    corners_struct = grid_cpp.get_cell_corners_from_ijk(*cell)

    assert isinstance(corners_struct, _internal.grid3d.CellCorners)
    corners = _internal.grid3d.arrange_corners(corners_struct)
    assert isinstance(corners, list)
    corners = np.array(corners).reshape((8, 3))
    assert np.allclose(corners[0], expected_firstcorner)
    assert np.allclose(corners[7], expected_lastcorner)


def test_cell_corners_minmax(testdata_path):
    """Test the cell_minmax function, which returns a list from C++."""
    # Read the banal6 grid

    cell = (0, 0, 0)
    grid = xtgeo.grid_from_file(f"{testdata_path}/3dgrids/etc/banal6.roff")
    grid_cpp = _internal.grid3d.Grid(grid)
    corners = grid_cpp.get_cell_corners_from_ijk(*cell)
    # Get the min and max of the first cell
    minmax = _internal.grid3d.get_corners_minmax(corners)

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
    grid_cpp = _internal.grid3d.Grid(grid)

    corners = grid_cpp.get_cell_corners_from_ijk(*cell)
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

    grid_cpp = _internal.grid3d.Grid(grid)

    corners = grid_cpp.get_cell_corners_from_ijk(*cell)
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


@functimer(output="info")
def test_get_cell_centers(get_drogondata):
    """Test cell centers from a grid; assertions are manually verified in RMS."""

    grid, _, _ = get_drogondata  # total cells 899944

    grid_cpp = _internal.grid3d.Grid(grid)
    xcor, ycor, zcor = grid_cpp.get_cell_centers(True)

    assert isinstance(xcor, np.ndarray)
    assert isinstance(ycor, np.ndarray)
    assert isinstance(zcor, np.ndarray)

    assert xcor.shape == (grid.ncol, grid.nrow, grid.nlay)

    assert np.nanmean(xcor) == pytest.approx(461461.6, abs=0.1)
    assert np.nanmean(ycor) == pytest.approx(5933128.2, abs=0.1)
    assert np.nanmean(zcor) == pytest.approx(1731.50, abs=0.1)

    assert np.nanstd(xcor) == pytest.approx(2260.1489, abs=0.1)
    assert np.nanstd(ycor) == pytest.approx(2945.992, abs=0.1)
    assert np.nanstd(zcor) == pytest.approx(67.27, abs=0.1)

    assert np.nansum(xcor) == pytest.approx(301686967304.8, abs=0.1)  # RMS 301686967112
    assert np.nansum(ycor) == pytest.approx(3878865619174.5, abs=0.1)  # 3878865622519
    assert np.nansum(zcor) == pytest.approx(1131994843.5, abs=0.1)  # 1131994843

    assert xcor[75, 23, 37] == pytest.approx(461837.19, abs=0.1)
    assert ycor[75, 23, 37] == pytest.approx(5937352, abs=0.1)
    assert zcor[75, 23, 37] == pytest.approx(1930.57, abs=0.1)

    assert np.isnan(xcor[62, 33, 37])
    assert np.isnan(ycor[62, 33, 37])
    assert np.isnan(zcor[62, 33, 37])


def test_process_edges_rmsapi(get_drogondata):
    """Test function that process boundary values."""

    grid, _, _ = get_drogondata

    zcv = grid._zcornsv.copy()

    # now manipulate pillar corner i=0, j=0, k=2 where the four numbers are equal.
    # Corner 0, 1, 2 are at edge and per se _outside_ the grid, while corner 3 (NE) is
    # inside. If I change corner 3, the routine should fix this so that the
    # remaining outside corners will have the same value
    #
    #      |
    #      |    CELL (0, 0, *)
    #      |
    #    2 | 3
    #    ------------------
    #    0 | 1

    # initially
    assert zcv[0, 0, 2, :].tolist() == pytest.approx(
        [1731.4475, 1731.4475, 1731.4475, 1731.4475]
    )

    # manipulate corner 3 (NorthEast, NE)
    zcv[0, 0, 2, 3] = 1730.0

    assert zcv[0, 0, 2, :].tolist() == pytest.approx(
        [1731.4475, 1731.4475, 1731.4475, 1730.0]
    )

    # process edges
    _internal.grid3d.process_edges_rmsapi(zcv)
    assert zcv[0, 0, 2, :].tolist() == pytest.approx([1730.0, 1730.0, 1730.0, 1730.0])


def test_convert_xtgeo_to_rmsapi(get_drogondata):
    """Test function that convert from xtgeo 3D grid to RMSAPI."""

    grid, _, _ = get_drogondata

    grid_cpp = _internal.grid3d.Grid(grid)
    tpillars, bpillars, zcorners, zmask = grid_cpp.convert_xtgeo_to_rmsapi()

    assert tpillars.all() == grid._coordsv[:, :, :3].all()
    assert bpillars.all() == grid._coordsv[:, :, 3:].all()

    assert np.all(zmask[0, 0, 0, :])
    assert np.all(zmask[grid.ncol, grid.nrow, 3, :])
    assert not np.all(zmask[grid.ncol, grid.nrow, 0, :])

    assert zcorners[10, 10, 0, :].all() == grid._zcornsv[10, 10, :, 0].all()


def test_convert_xtgeo_to_rmsapi_warnings(get_drogondata):
    """Test warnings when convert from xtgeo 3D grid to RMSAPI."""

    grid, _, _ = get_drogondata

    # manipulate grid values
    use_grid = grid.copy()
    use_grid._coordsv[10, 10, 2] = use_grid._coordsv[10, 10, 5] = 1999.0

    grid_cpp = _internal.grid3d.Grid(use_grid)

    with pytest.warns(UserWarning, match="Equal Z coordinates detected"):
        grid_cpp.convert_xtgeo_to_rmsapi()

    # crossing zcoords
    use_grid = grid.copy()

    use_grid._zcornsv[10, 10, 5, 0] = 1999.9999

    grid_cpp = _internal.grid3d.Grid(use_grid)
    with pytest.warns(UserWarning, match="One or more ZCORN values are crossing"):
        grid_cpp.convert_xtgeo_to_rmsapi()


def test_adjust_box_grid_to_regsurfs():
    """Test the adjust_box_grid function, which updates zcorns in the grid."""
    # Create a simple grid
    grid = xtgeo.create_box_grid((8, 4, 5))

    assert grid._zcornsv[4, 2, :, 1].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    surf = xtgeo.surface_from_grid3d(grid, where="top")
    surfaces = []
    for i in range(6):
        surfn = surf.copy()
        surfn.values += 2 * i
        surf_cpp = _internal.regsurf.RegularSurface(surfn)
        surfaces.append(surf_cpp)

    grd_cpp = _internal.grid3d.Grid(grid)
    new_zcorns, _active = grd_cpp.adjust_boxgrid_layers_from_regsurfs(surfaces)
    assert new_zcorns[4, 2, :, 1].tolist() == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]


def test_extract_onelayer_grid(get_drogondata):
    """get a grid with one layer"""

    @functimer(output="print")
    def _create_grid(get_drogondata):
        grid, _, _ = get_drogondata

        grid_cpp = _internal.grid3d.Grid(grid)

        return grid_cpp.extract_onelayer_grid()

    new_grid_cpp = _create_grid(get_drogondata)

    assert isinstance(new_grid_cpp, xtgeo._internal.grid3d.Grid)

    # now make into Python; not needed bit for eventual QC
    grd = xtgeo.Grid(
        coordsv=new_grid_cpp.coordsv,
        zcornsv=new_grid_cpp.zcornsv,
        actnumsv=new_grid_cpp.actnumsv,
    )
    assert isinstance(grd, xtgeo.Grid)
    assert grd.nlay == 1
    assert grd.ncol == 92
    assert grd.nrow == 146


def test_get_grid_boundingbox(get_drogondata):
    """Test the get_bounding_box function"""

    grid, _, _ = get_drogondata

    grid_cpp = _internal.grid3d.Grid(grid)

    pmin, pmax = grid_cpp.get_bounding_box()

    assert (pmin.x, pmin.y, pmin.z) == pytest.approx(
        (456063.6875, 5926551.0, 1554.1617431640625)
    )

    assert (pmax.x, pmax.y, pmax.z) == pytest.approx(
        (467489.34375, 5939441.0, 2001.5931396484375)
    )
