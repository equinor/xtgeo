import pathlib

import numpy as np
import pytest

import xtgeo
from xtgeo.common.log import functimer

REEKROOT = pathlib.Path("3dgrids/reek/REEK")
WELL1 = pathlib.Path("wells/reek/1/OP_1.w")
FENCE1 = pathlib.Path("polygons/reek/1/fence.pol")
FENCE2 = pathlib.Path("polygons/reek/1/minifence.pol")


@pytest.mark.parametrize("grid_kind", ["merged", "single-cell", "inactive-single-cell"])
@pytest.mark.parametrize(
    "ijk, center, radius, expected, nest_id",
    [
        (
            (24, 6, 8),
            (459609.7075219, 5937028.100338835, 1756.7418327331543),
            0.03,
            962.6852416992188,
            1,
        ),
        (
            (64, 3, 64),
            (459780.2179552294, 5937106.740874652, 1784.8945808410645),
            0.0003,
            1076.680419921875,
            2,
        ),
        (
            (48, 23, 2),
            (459585.12565492745, 5935208.380217581, 1631.6068241882324),
            0.0003,
            988.5105590820312,
            2,
        ),
    ],
    ids=["coarse", "refined", "refined-wrong-layer"],
)
def test_randomline_drogon_warped_cell(
    testdata_path, grid_kind, ijk, center, radius, expected, nest_id
):
    """Sample interior points in warped coarse and refined cells.

    The same geometry must work in the merged grid and as a single cell.
    Missing samples must remain undefined if that cell is inactive.
    """
    path = pathlib.Path(testdata_path) / "3dgrids/drogon/5/drogon_nested_hybrid1.roff"
    grid = xtgeo.grid_from_file(path)
    prop = xtgeo.gridproperty_from_file(path, name="KX", grid=grid)
    nest = xtgeo.gridproperty_from_file(path, name="NEST_ID", grid=grid)
    index = tuple(k - 1 for k in ijk)
    assert grid.get_actnum().values[index] == 1
    assert nest.values[index] == nest_id
    assert prop.values[index] == pytest.approx(expected)
    if grid_kind != "merged":
        grid.crop(*[(k, k) for k in ijk])
        prop = xtgeo.GridProperty(grid, values=expected)
    if grid_kind == "inactive-single-cell":
        actnum = grid.get_actnum()
        actnum.values[:] = 0
        grid.set_actnum(actnum)

    npoints = 7 if grid_kind == "merged" else 5
    offsets = np.linspace(-radius, radius, npoints)
    fence = np.column_stack(
        (
            center[0] + offsets,
            np.full(npoints, center[1]),
            np.zeros(npoints),
            offsets - offsets[0],
        )
    )
    *_, values = grid.get_randomline(
        fence,
        prop,
        zmin=center[2] - radius,
        zmax=center[2] + radius,
        # Keep seven depth samples despite rounding in the interval calculation.
        zincrement=2 * radius / 6.00001,
    )
    assert values.shape == (7, npoints)
    if grid_kind == "inactive-single-cell":
        assert np.isnan(values).all()
    else:
        np.testing.assert_allclose(values, expected)


@pytest.mark.parametrize("inactive", [False, True])
def test_randomline_regular_grid_values_and_undefined_samples(inactive):
    """Return known box-cell values and NaN outside its lateral and depth bounds."""
    grid = xtgeo.create_box_grid((2, 2, 2))
    ii, jj, kk = np.indices(grid.dimensions)
    prop = xtgeo.GridProperty(grid, values=ii + 2 * jj + 4 * kk)
    if inactive:
        actnum = grid.get_actnum()
        actnum.values[1, 0, 0] = 0
        grid.set_actnum(actnum)
    xs = np.arange(-0.25, 3, 0.5)
    fence = np.column_stack((xs, np.full(7, 0.5), np.zeros(7), xs - xs[0]))
    *_, values = grid.get_randomline(fence, prop, zmin=-0.25, zmax=2.25, zincrement=0.5)
    expected = np.full((6, 7), np.nan)
    expected[1:3, 1:5] = [0, 0, 1, 1]
    expected[3:5, 1:5] = [4, 4, 5, 5]
    if inactive:
        expected[1:3, 3:5] = np.nan
    np.testing.assert_array_equal(values, expected)


def test_randomline_prefers_refined_cell_over_inactive_parent(testdata_path):
    """Return the active refined value when an inactive parent also contains a point."""
    path = pathlib.Path(testdata_path) / "3dgrids/drogon/5/drogon_nested_hybrid1.roff"
    fine = xtgeo.grid_from_file(path)
    fine.crop((64, 64), (3, 3), (64, 64))
    corners = np.asarray(fine.get_xyz_cell_corners()).reshape(8, 3)
    low, high = corners.min(axis=0), corners.max(axis=0)
    parent = xtgeo.create_box_grid(
        (1, 1, 1), origin=tuple(low - 1), increment=tuple(high - low + 2)
    )
    actnum = parent.get_actnum()
    actnum.values[:] = 0
    parent.set_actnum(actnum)
    grid = xtgeo.grid_merge(parent, fine)
    prop = xtgeo.GridProperty(grid, values=10 + 10 * np.indices(grid.dimensions)[0])
    corners = np.asarray(grid.get_xyz_cell_corners((3, 1, 1))).reshape(8, 3)
    weights = np.array([9, 3, 3, 1]) / 16
    center = (weights @ corners[:4]) * (1 - 1e-5) + (weights @ corners[4:]) * 1e-5
    offsets = np.linspace(-1e-7, 1e-7, 7)
    fence = np.column_stack(
        (center[0] + offsets, np.full(7, center[1]), np.zeros(7), offsets - offsets[0])
    )
    *_, values = grid.get_randomline(
        fence,
        prop,
        zmin=center[2] - 1e-7,
        zmax=center[2] + 1e-7,
        zincrement=2e-7 / 6.00001,
    )
    np.testing.assert_allclose(values, 30)


def test_randomline_fence_from_well(show_plot, testdata_path):
    """Import ROFF grid with props and make fences"""

    grd = xtgeo.grid_from_file(
        testdata_path / REEKROOT, fformat="eclipserun", initprops=["PORO"]
    )
    wll = xtgeo.well_from_file(testdata_path / WELL1, zonelogname="Zonelog")

    # get the polygon for the well, limit it to 1200
    fspec = wll.get_fence_polyline(sampling=10, nextend=2, asnumpy=False, tvdmin=100)

    # assert fspec.get_dataframe()[fspec.dhname][4] == pytest.approx(12.6335, abs=0.001)

    @functimer(output="print")
    def get_polyline():
        """Get the fence polyline as a numpy array."""
        return wll.get_fence_polyline(sampling=1, nextend=10, asnumpy=True, tvdmin=1200)

    fspec = get_polyline()

    # get the "image", which is a 2D numpy that can be plotted with e.g. imgshow

    @functimer(output="print")
    def get_randomline():
        hmin, hmax, vmin, vmax, por = grd.get_randomline(
            fspec, "PORO", zmin=1550, zmax=1660, zincrement=0.1
        )
        return hmin, hmax, vmin, vmax, por

    hmin, hmax, vmin, vmax, por = get_randomline()
    hmin, hmax, vmin, vmax, por = get_randomline()

    @functimer(output="print")
    def showplot():
        import matplotlib.pyplot as plt

        plt.figure()
        plt.imshow(por, cmap="rainbow", extent=(hmin, hmax, vmax, vmin))
        plt.axis("tight")
        plt.colorbar()

        return plt

    if show_plot:
        plt = showplot()
        plt.show()


def test_randomline_fence_from_polygon(show_plot, testdata_path):
    """Import ROFF grid with props and make fence from polygons"""

    grd = xtgeo.grid_from_file(
        testdata_path / REEKROOT, fformat="eclipserun", initprops=["PORO", "PERMX"]
    )
    fence = xtgeo.polygons_from_file(testdata_path / FENCE1)

    # get the polygons
    fspec = fence.get_fence(distance=10, nextend=2, asnumpy=False)
    assert fspec.get_dataframe()[fspec.dhname][4] == pytest.approx(10, abs=1)

    fspec = fence.get_fence(distance=5, nextend=2, asnumpy=True)

    # get the "image", which is a 2D numpy that can be plotted with e.g. imgshow
    hmin, hmax, vmin, vmax, por = grd.get_randomline(
        fspec, "PORO", zmin=1680, zmax=1750, zincrement=0.5
    )

    hmin, hmax, vmin, vmax, perm = grd.get_randomline(
        fspec, "PERMX", zmin=1680, zmax=1750, zincrement=0.5
    )

    if show_plot:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.imshow(por, cmap="rainbow", extent=(hmin, hmax, vmax, vmin))
        plt.axis("tight")
        plt.colorbar()
        plt.figure()
        plt.imshow(perm, cmap="rainbow", extent=(hmin, hmax, vmax, vmin))
        plt.axis("tight")
        plt.colorbar()
        plt.show()


def test_randomline_fence_calczminzmax(show_plot, testdata_path):
    """Import ROFF grid with props and make fence from polygons, zmin/zmax auto"""

    grd = xtgeo.grid_from_file(
        testdata_path / REEKROOT, fformat="eclipserun", initprops=["PORO", "PERMX"]
    )
    fence = xtgeo.polygons_from_file(testdata_path / FENCE1)

    fspec = fence.get_fence(distance=5, nextend=2, asnumpy=True)

    hmin, hmax, vmin, vmax, xx = grd.get_randomline(fspec, "PORO", zmin=None, zmax=None)
    assert vmin == pytest.approx(1547.98, abs=0.01)
    if show_plot:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.imshow(xx, cmap="rainbow", extent=(hmin, hmax, vmax, vmin))
        plt.axis("tight")
        plt.colorbar()
        plt.show()
