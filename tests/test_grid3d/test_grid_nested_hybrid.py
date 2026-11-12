import pathlib

import numpy as np
import pytest

import xtgeo
from xtgeo.grid3d import _grid_nested_hybrid


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("3dgrids/drogon/5/drogon_nested_hybrid1.roff", True),
        ("3dgrids/eme/3/nested_grid_refine_2_2_2_merged.grdecl", True),
        ("3dgrids/eme/3/onecellregion1_refine_2_2_2_merged.grdecl", True),
        ("3dgrids/eme/3/original.grdecl", False),
        ("3dgrids/reek/REEK.EGRID", False),
    ],
)
def test_imported_grid(testdata_path, filename, expected):
    grid = xtgeo.grid_from_file(pathlib.Path(testdata_path) / filename)
    assert grid.nestedhybrid is expected


@pytest.mark.parametrize("fformat", ["roff", "grdecl", "bgrdecl", "egrid"])
def test_round_trip(nested_box, tmp_path, fformat):
    nested_grid = nested_box[1]
    assert nested_grid.nestedhybrid
    path = tmp_path / f"nested.{fformat}"
    nested_grid.to_file(path, fformat=fformat)
    assert xtgeo.grid_from_file(path, fformat=fformat).nestedhybrid


@pytest.mark.parametrize("empty", [[], [0], [3], [2, 3], [1, 3], list(range(6))])
def test_inactive_columns_are_not_enough(empty):
    grid = xtgeo.create_box_grid((6, 4, 2))
    actnum = grid.get_actnum()
    actnum.values[empty, :, :] = 0
    grid.set_actnum(actnum)
    assert not grid.nestedhybrid


def test_disjoint_merge_is_rejected_before_centres(monkeypatch):
    coarse = xtgeo.create_box_grid((4, 4, 2))
    other = xtgeo.create_box_grid((2, 2, 2), origin=(10, 0, 0))
    merged = xtgeo.grid_merge(coarse, other)

    def fail(*args):
        raise AssertionError("disjoint bounds should be rejected first")

    monkeypatch.setattr(_grid_nested_hybrid, "_active_xy", fail)
    assert merged.nestedhybrid is False


@pytest.mark.parametrize("leading", [1, 2])
def test_leading_inactive_columns(nested_box, leading):
    nested_grid = nested_box[1]
    actnum = nested_grid.get_actnum()
    actnum.values[:leading, :, :] = 0
    nested_grid.set_actnum(actnum)
    assert nested_grid.nestedhybrid


def test_detection_is_cached(nested_box, monkeypatch):
    nested_grid = nested_box[1]
    detect = _grid_nested_hybrid.detect_nested_hybrid
    calls = 0

    def count_calls(grid):
        nonlocal calls
        calls += 1
        return detect(grid)

    monkeypatch.setattr(_grid_nested_hybrid, "detect_nested_hybrid", count_calls)
    assert calls == 0
    assert nested_grid.nestedhybrid
    assert nested_grid.nestedhybrid
    assert calls == 1

    nested_grid._coordsv[5:, :, [0, 3]] += 10000
    assert not nested_grid.nestedhybrid
    assert calls == 2


def test_centres_only_for_first_refined_block(nested_box, monkeypatch):
    nested_grid = nested_box[1]
    further = xtgeo.create_box_grid((20, 4, 4), origin=(10, 0, 0))
    merged = xtgeo.grid_merge(nested_grid, further)
    active_xy = _grid_nested_hybrid._active_xy
    dimensions = []

    def record_centres(coordsv, zcornsv, active):
        dimensions.append(active.shape)
        return active_xy(coordsv, zcornsv, active)

    monkeypatch.setattr(_grid_nested_hybrid, "_active_xy", record_centres)
    assert merged.nestedhybrid
    assert dimensions == [(4, 4, 4)]


@pytest.mark.parametrize("geometry", ["inclined", "extrapolated", "collapsed"])
def test_active_xy_matches_cell_centres(geometry):
    grid = xtgeo.create_box_grid((4, 4, 3), origin=(450000, 5900000, 0))
    grid._set_xtgformat2()
    grid._coordsv[:, :, 3] += 20
    if geometry == "extrapolated":
        grid._zcornsv += 10
    elif geometry == "collapsed":
        grid._coordsv[:, :, 5] = grid._coordsv[:, :, 2]
    grid._actnumsv[::2, ::2, :] = 0
    active = grid._actnumsv > 0
    px, py = _grid_nested_hybrid._active_xy(grid._coordsv, grid._zcornsv, active)
    x, y, _ = grid.get_xyz()
    np.testing.assert_allclose(px, x.values.data[active], rtol=0, atol=1e-8)
    np.testing.assert_allclose(py, y.values.data[active], rtol=0, atol=1e-8)


def test_extrapolated_fine_pillars():
    coarse = xtgeo.create_box_grid((4, 4, 2), origin=(49, 0, 0))
    refined = xtgeo.create_box_grid((1, 1, 1), origin=(0, 1, 0))
    refined._set_xtgformat2()
    refined._coordsv[:, :, 3] += 20
    refined._zcornsv += 2
    assert xtgeo.grid_merge(coarse, refined).nestedhybrid


def test_rotated_nested():
    coarse = xtgeo.create_box_grid((10, 10, 2), rotation=30)
    refined = coarse.copy()
    refined.crop((4, 6), (4, 6), (1, 2))
    refined.refine(2, 2, 1)
    actnum = coarse.get_actnum()
    actnum.values[3:6, 3:6, :] = 0
    coarse.set_actnum(actnum)
    assert xtgeo.grid_merge(coarse, refined).nestedhybrid


def test_rotated_corner_is_not_nested():
    coarse = xtgeo.create_box_grid((10, 10, 2), rotation=45)
    refined = xtgeo.create_box_grid(
        (2, 2, 2), origin=(5, 0.5, 0), increment=(0.5, 0.5, 1)
    )
    assert not xtgeo.grid_merge(coarse, refined).nestedhybrid


def test_concave_recess_is_not_nested():
    coarse = xtgeo.create_box_grid((2, 2, 1))
    coarse._set_xtgformat2()
    coarse._coordsv[:, 1, [1, 4]] = 0.25
    coarse._coordsv[1, 2, [1, 4]] = 0.75
    refined = xtgeo.create_box_grid(
        (1, 1, 1), origin=(0.9, 1.4, 0), increment=(0.2, 0.2, 1)
    )
    assert not xtgeo.grid_merge(coarse, refined).nestedhybrid


@pytest.mark.parametrize(
    "origin, bottom_shift, expected",
    [
        ((0.8, 0.8, 0), 0, False),
        ((0.8, 0.8, 0), 0.01, False),
        ((0.1, 0.1, 0), 0, True),
    ],
)
def test_single_concave_cell(origin, bottom_shift, expected):
    coarse = xtgeo.create_box_grid((1, 1, 1))
    coarse._set_xtgformat2()
    coarse._coordsv[1, 0, [0, 3]] = 2
    coarse._coordsv[0, 1, [1, 4]] = 2
    coarse._coordsv[1, 1, :] = (0.5, 0.5, 0, 0.5, 0.5, 1)
    coarse._coordsv[:, :, 3] += bottom_shift
    refined = xtgeo.create_box_grid((1, 1, 1), origin=origin, increment=(0.2, 0.2, 1))
    assert xtgeo.grid_merge(coarse, refined).nestedhybrid is expected


@pytest.mark.parametrize("shift, depth", [(0, 0), (10, 500)])
def test_taller_third_grid_does_not_change_detection(shift, depth):
    coarse = xtgeo.create_box_grid((4, 4, 1))
    coarse._set_xtgformat2()
    coarse._coordsv[..., 3] += shift
    fine = xtgeo.create_box_grid((2, 2, 1), origin=(3, 5, 0))
    merged = xtgeo.grid_merge(coarse, fine)
    assert not merged.nestedhybrid

    taller = xtgeo.create_box_grid((1, 20, 1), origin=(1000, 1000, depth))
    extended = xtgeo.grid_merge(merged, taller)
    assert not extended.nestedhybrid


def test_multiple_column_gap(nested_box):
    nested_grid = nested_box[1]
    nested_grid._set_xtgformat2()
    grid = xtgeo.Grid(
        coordsv=np.insert(nested_grid._coordsv, 4, nested_grid._coordsv[4], axis=0),
        zcornsv=np.insert(nested_grid._zcornsv, 4, nested_grid._zcornsv[4], axis=0),
        actnumsv=np.insert(nested_grid._actnumsv, 4, 0, axis=0),
    )
    assert grid.nestedhybrid


def test_multiple_strips_in_nested_layout(nested_box):
    nested_grid = nested_box[1]
    actnum = nested_grid.get_actnum()
    actnum.values[1, :, :] = 0
    nested_grid.set_actnum(actnum)
    assert not nested_grid.nestedhybrid


@pytest.mark.parametrize("origin", [(1, 1, 0), (10, 0, 0)])
def test_multi_level(nested_box, origin):
    nested_grid = nested_box[1]
    further = xtgeo.create_box_grid((2, 2, 2), origin=origin)
    assert xtgeo.grid_merge(nested_grid, further).nestedhybrid


def test_multi_level_checks_first_pair_only():
    coarse = xtgeo.create_box_grid((4, 4, 2))
    disjoint = xtgeo.create_box_grid((2, 2, 2), origin=(10, 0, 0))
    refined = xtgeo.create_box_grid((2, 2, 2), origin=(1, 1, 0))
    merged = xtgeo.grid_merge(xtgeo.grid_merge(coarse, disjoint), refined)
    assert not merged.nestedhybrid


def test_trailing_inactive_strip(nested_box):
    nested_grid = nested_box[1]
    inactive = xtgeo.create_box_grid((2, 2, 2), origin=(10, 0, 0))
    actnum = inactive.get_actnum()
    actnum.values[:] = 0
    inactive.set_actnum(actnum)
    assert xtgeo.grid_merge(nested_grid, inactive).nestedhybrid


@pytest.mark.parametrize("depth_shift", [0, 2, -2])
def test_inclined_pillars(depth_shift):
    coarse = xtgeo.create_box_grid((4, 4, 2))
    coarse._set_xtgformat2()
    coarse._coordsv[:, :, 3] += 20
    coarse._zcornsv += depth_shift
    refined = coarse.copy()
    refined.crop((2, 3), (2, 3), (1, 2))
    assert xtgeo.grid_merge(coarse, refined).nestedhybrid


@pytest.mark.parametrize("offset, expected", [(0, True), (5, False)])
def test_collapsed_coarse_columns(offset, expected):
    coarse = xtgeo.create_box_grid((4, 4, 2))
    coarse._set_xtgformat2()
    coarse._coordsv[:, :, 3] += 1
    refined = coarse.copy()
    refined.crop((2, 3), (2, 3), (1, 2))
    refined._set_xtgformat2()
    refined._coordsv[:, :, [0, 3]] += offset
    actnum = coarse.get_actnum()
    actnum.values[1:3, 1:3, :] = 0
    coarse.set_actnum(actnum)
    # Flatten only the removed columns' quadrants on the shared pillars.
    for di, dj, quadrant in [(0, 0, 3), (1, 0, 2), (1, 1, 0), (0, 1, 1)]:
        coarse._zcornsv[1 + di : 3 + di, 1 + dj : 3 + dj, :, quadrant] = -10
    assert xtgeo.grid_merge(coarse, refined).nestedhybrid is expected


@pytest.mark.parametrize("inside, expected", [(10, True), (9, False)])
def test_all_centres_must_be_inside(inside, expected):
    coarse = xtgeo.create_box_grid((inside, 2, 2))
    refined = xtgeo.create_box_grid((10, 2, 2))
    assert xtgeo.grid_merge(coarse, refined).nestedhybrid is expected


def test_active_collapsed_column_keeps_its_depths():
    coarse = xtgeo.create_box_grid((1, 1, 1))
    coarse._set_xtgformat2()
    coarse._coordsv[..., 3] += 10
    coarse._zcornsv[:] = 0
    refined = xtgeo.create_box_grid((1, 1, 1), origin=(10, 0, 1))
    assert not xtgeo.grid_merge(coarse, refined).nestedhybrid


def test_two_later_blocks_do_not_change_first_pair():
    coarse = xtgeo.create_box_grid((4, 4, 1))
    pair = xtgeo.grid_merge(coarse, xtgeo.create_box_grid((2, 2, 1), (3, 5, 0)))
    third = xtgeo.create_box_grid((1, 20, 1), origin=(1000, 1000, 0))
    fourth = xtgeo.create_box_grid((1, 30, 1), origin=(2000, 2000, 0))
    merged = xtgeo.grid_merge(xtgeo.grid_merge(pair, third), fourth)
    assert not merged.nestedhybrid


def test_empty_grid_is_not_nested():
    assert xtgeo.create_box_grid((0, 0, 0)).nestedhybrid is False


@pytest.mark.parametrize("inactive", [False, True])
def test_pinched_coarse_boundary_is_kept(inactive):
    coarse = xtgeo.create_box_grid((1, 2, 1))
    coarse._set_xtgformat2()
    coarse._coordsv[1, 0, [0, 3]] = 0
    if inactive:
        actnum = coarse.get_actnum()
        actnum.values[0, 0, 0] = 0
        coarse.set_actnum(actnum)
    refined = xtgeo.create_box_grid(
        (1, 1, 1), origin=(0.1, 0.5, 0), increment=(0.2, 0.2, 1)
    )
    assert xtgeo.grid_merge(coarse, refined).nestedhybrid is True


def test_grid_ending_in_inactive_strip_has_no_fine_block():
    grid = xtgeo.create_box_grid((4, 2, 1))
    actnum = grid.get_actnum()
    actnum.values[2:, :, :] = 0
    grid.set_actnum(actnum)
    assert grid.nestedhybrid is False


def test_nestedhybrid_blocks_locate_every_level(nested_box):
    coarse, merged = nested_box
    further = xtgeo.create_box_grid(
        (2, 2, 2), origin=(150, 150, 1520), increment=(25, 25, 10)
    )
    assert merged.nestedhybrid_blocks == merged.nestedhybrid_blocks
    blocks = xtgeo.grid_merge(merged, further).nestedhybrid_blocks

    assert [(b.istart, b.istop) for b in blocks] == [(0, 4), (5, 9), (10, 12)]


def test_nestedhybrid_blocks_empty_when_not_nested():
    assert xtgeo.create_box_grid((3, 3, 1)).nestedhybrid_blocks == []
