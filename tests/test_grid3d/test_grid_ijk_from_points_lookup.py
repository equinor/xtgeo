"""Tests for ``Grid.get_ijk_from_points`` lookup engines.

Two engines resolve a point to its enclosing cell:

* ``"guided"`` (default) narrows the search with guide surfaces and falls back to a
  spatial-hash index for points it cannot resolve that way.
* ``"spatial"`` uses the same spatial-hash index for every point, cached on the grid.

Both must agree, resolve nested hybrid grids (where a refined cell's ``(I, J)`` is
far from the coarse "mother" cells it is geometrically nested inside), and prefer
an active cell over an overlapping inactive one.
"""

import logging
import pathlib

import numpy as np
import pytest

import xtgeo

NESTEDHYBRID1 = pathlib.Path("3dgrids/drogon/5/drogon_nested_hybrid1.roff")

METHODS = ("guided", "spatial")


@pytest.fixture(scope="module")
def nested_hybrid_grid(testdata_path):
    """Load the Drogon nested hybrid grid and its NEST_ID property."""
    path = pathlib.Path(testdata_path) / NESTEDHYBRID1
    grid = xtgeo.grid_from_file(path)
    nest = xtgeo.gridproperty_from_file(path, name="NEST_ID", grid=grid)
    return grid, nest


def _sample_cell_centers(grid, mask, n, seed):
    """Return (Points, truth_ijk) for ``n`` random active cells within ``mask``."""
    xx, yy, zz = grid.get_xyz(asmasked=True)
    valid = mask & ~np.ma.getmaskarray(xx.values)
    ii, jj, kk = np.where(valid)
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(ii), size=min(n, len(ii)), replace=False)
    i, j, k = ii[sel], jj[sel], kk[sel]
    points = np.column_stack(
        [
            np.asarray(xx.values)[i, j, k],
            np.asarray(yy.values)[i, j, k],
            np.asarray(zz.values)[i, j, k],
        ]
    )
    truth = np.column_stack([i + 1, j + 1, k + 1])
    return xtgeo.Points(list(map(tuple, points))), truth


def _box_grid_centers(grid):
    """Return (Points, truth_ijk) for every cell center of a box grid."""
    xx, yy, zz = grid.get_xyz(asmasked=True)
    idx = np.indices(grid.dimensions).reshape(3, -1).T
    points = np.column_stack([np.asarray(a.values)[tuple(idx.T)] for a in (xx, yy, zz)])
    return xtgeo.Points(list(map(tuple, points))), idx + 1


def _ijk(grid, points, **kwargs):
    return grid.get_ijk_from_points(points, **kwargs)[["IX", "JY", "KZ"]].to_numpy()


def _index_builds(caplog):
    """Count how many times the C++ layer built a spatial index."""
    return sum("Building spatial index" in rec.message for rec in caplog.records)


def test_nested_hybrid_layout(nested_hybrid_grid):
    """Refined cells overlap the mother footprint but use distant I/J indices."""
    grid, nest = nested_hybrid_grid
    refined = (nest.values == 2).filled(False)
    mother = (nest.values == 1).filled(False)
    assert np.where(refined)[0].min() > np.where(mother)[0].max()
    xx, yy, _ = grid.get_xyz(asmasked=True)
    assert xx.values[mother].min() <= xx.values[refined].min()
    assert xx.values[refined].max() <= xx.values[mother].max()
    assert yy.values[mother].min() <= yy.values[refined].min()
    assert yy.values[refined].max() <= yy.values[mother].max()


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(("nest_id", "seed"), [(2, 42), (1, 1)])
def test_nested_cell_centers_resolve(nested_hybrid_grid, method, nest_id, seed):
    """Mother and refined cell centers resolve to their exact active cells."""
    grid, nest = nested_hybrid_grid
    cells = (nest.values == nest_id).filled(False)
    points, truth = _sample_cell_centers(grid, cells, 500, seed=seed)

    got = _ijk(grid, points, activeonly=True, method=method)

    np.testing.assert_array_equal(got, truth)
    returned_nest = nest.values.filled(0)[got[:, 0] - 1, got[:, 1] - 1, got[:, 2] - 1]
    assert np.all(returned_nest == nest_id)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("activeonly", [True, False])
def test_regular_box_grid(method, activeonly):
    """Both engines are exact on a regular structured grid."""
    grid = xtgeo.create_box_grid(
        (5, 4, 3), increment=(100, 100, 10), origin=(0, 0, 1000)
    )
    points, truth = _box_grid_centers(grid)

    got = _ijk(grid, points, activeonly=activeonly, method=method)
    np.testing.assert_array_equal(got, truth)


def test_spatial_cache_reused_across_calls(nested_hybrid_grid, caplog):
    """A second spatial call reuses the cached index instead of rebuilding it."""
    grid, nest = nested_hybrid_grid
    refined = (nest.values == 2).filled(False)
    points, truth = _sample_cell_centers(grid, refined, 300, seed=11)
    grid = grid.copy()  # fresh C++ grid, so the index cache starts out empty

    with caplog.at_level(logging.DEBUG, logger="build_spatial_index"):
        first = _ijk(grid, points, activeonly=True, method="spatial")
        assert _index_builds(caplog) == 1, "index was not built on the first call"
        second = _ijk(grid, points, activeonly=True, method="spatial")
        assert _index_builds(caplog) == 1, "index was rebuilt instead of reused"

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first, truth)


def test_spatial_cache_rebuilt_when_activeonly_changes(nested_hybrid_grid, caplog):
    """The cached index is tied to activeonly, so flipping it forces a rebuild."""
    grid, nest = nested_hybrid_grid
    refined = (nest.values == 2).filled(False)
    points, _ = _sample_cell_centers(grid, refined, 50, seed=3)
    grid = grid.copy()  # fresh C++ grid, so the index cache starts out empty

    with caplog.at_level(logging.DEBUG, logger="build_spatial_index"):
        _ijk(grid, points, activeonly=True, method="spatial")
        _ijk(grid, points, activeonly=False, method="spatial")
        assert _index_builds(caplog) == 2


def test_empty_pointset_does_not_build_index(caplog):
    """A no-op query returns empty results without paying for the index."""
    grid = xtgeo.create_box_grid(
        (4, 4, 3), increment=(100, 100, 10), origin=(0, 0, 1000)
    )
    points = xtgeo.Points(np.empty((0, 3)))

    with caplog.at_level(logging.DEBUG, logger="build_spatial_index"):
        got = _ijk(grid, points, activeonly=True, method="spatial")

    assert got.shape[0] == 0
    assert _index_builds(caplog) == 0


def test_empty_grid_spatial_returns_undefined_indices():
    """A spatial query against a grid without cells returns undefined indices."""
    grid = xtgeo.create_box_grid((0, 0, 0))
    points = xtgeo.Points([(0.0, 0.0, 0.0), (10.0, 10.0, 10.0)])

    got = _ijk(grid, points, activeonly=True, method="spatial")

    np.testing.assert_array_equal(got, np.full((2, 3), -1))


def test_single_point_query_matches_between_engines():
    """A one-point query builds the index correctly and agrees with guided."""
    grid = xtgeo.create_box_grid(
        (4, 4, 2), increment=(100, 100, 10), origin=(0, 0, 1000)
    )
    xx, yy, zz = grid.get_xyz(asmasked=True)
    points = xtgeo.Points(
        [
            (
                float(xx.values[0, 0, 0]),
                float(yy.values[0, 0, 0]),
                float(zz.values[0, 0, 0]),
            )
        ]
    )

    spatial = _ijk(grid, points, activeonly=False, method="spatial")
    guided = _ijk(grid, points, activeonly=False, method="guided")

    np.testing.assert_array_equal(spatial, guided)
    np.testing.assert_array_equal(spatial, np.array([(1, 1, 1)]))


def test_invalid_method_raises():
    grid = xtgeo.create_box_grid((2, 2, 2))
    points = xtgeo.Points([(0.0, 0.0, 0.0)])
    with pytest.raises(ValueError, match="Unknown method"):
        grid.get_ijk_from_points(points, method="bogus")


@pytest.mark.parametrize("method", METHODS)
def test_inactive_only_cell_is_reported_when_not_activeonly(method):
    """With activeonly=False a lone inactive cell is still reported."""
    grid = xtgeo.create_box_grid(
        (4, 4, 3), increment=(100, 100, 10), origin=(0, 0, 1000)
    )
    actnum = grid.get_actnum()
    values = actnum.values.copy()
    values[1, 1, 1] = 0
    actnum.values = values
    grid.set_actnum(actnum)

    xx, yy, zz = grid.get_xyz(asmasked=False)
    points = xtgeo.Points(
        [
            (
                float(xx.values[1, 1, 1]),
                float(yy.values[1, 1, 1]),
                float(zz.values[1, 1, 1]),
            )
        ]
    )

    np.testing.assert_array_equal(
        _ijk(grid, points, activeonly=True, method=method), np.array([(-1, -1, -1)])
    )
    np.testing.assert_array_equal(
        _ijk(grid, points, activeonly=False, method=method), np.array([(2, 2, 2)])
    )


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_points_are_undefined(method, bad):
    """NaN/inf coordinates report -1 instead of an arbitrary cell."""
    grid = xtgeo.create_box_grid(
        (4, 4, 3), increment=(100, 100, 10), origin=(0, 0, 1000)
    )
    points = xtgeo.Points([(150.0, 150.0, 1005.0), (bad, 150.0, 1005.0)])

    got = _ijk(grid, points, activeonly=False, method=method)
    np.testing.assert_array_equal(got[1], np.array([-1, -1, -1]))
    assert got[0][0] > 0


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("activeonly", [True, False])
def test_fortran_ordered_actnum_matches_c_order(method, activeonly):
    """ACTNUM is read by cell number, so the constructor normalizes the layout.

    A Fortran-ordered array would otherwise be read as if it were C-ordered and
    silently point at the wrong cells.
    """
    grid = xtgeo.create_box_grid(
        (4, 3, 2), increment=(100, 100, 10), origin=(0, 0, 1000)
    )
    actnum = grid._actnumsv.copy()
    actnum[1, 0, 0] = 0  # asymmetric, so C and Fortran order disagree
    c_grid = xtgeo.Grid(grid._coordsv, grid._zcornsv, np.ascontiguousarray(actnum))
    f_grid = xtgeo.Grid(grid._coordsv, grid._zcornsv, np.asfortranarray(actnum))
    assert f_grid._actnumsv.flags["C_CONTIGUOUS"], "constructor did not normalize"

    xx, yy, zz = grid.get_xyz(asmasked=False)
    points = xtgeo.Points(
        np.column_stack([xx.values.ravel(), yy.values.ravel(), zz.values.ravel()])
    )

    expected = _ijk(c_grid, points, activeonly=activeonly, method=method)
    got = _ijk(f_grid, points, activeonly=activeonly, method=method)
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("method", METHODS)
def test_activeonly_false_prefers_active_nested_cell(nested_hybrid_grid, method):
    """Overlapping mother/refined cells resolve to the active refined cell."""
    grid, nest = nested_hybrid_grid
    refined = (nest.values == 2).filled(False)
    points, truth = _sample_cell_centers(grid, refined, 300, seed=7)

    np.testing.assert_array_equal(
        _ijk(grid, points, activeonly=False, method=method), truth
    )
