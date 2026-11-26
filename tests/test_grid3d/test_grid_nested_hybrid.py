import pathlib

import numpy as np
import pytest
import shapely

import xtgeo
from xtgeo.grid3d import _grid_nested_hybrid
from xtgeo.grid3d._grid_nested_hybrid import NestedHybridCellType


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


@pytest.mark.parametrize(
    "call",
    [
        lambda g: g.get_cell_transmissibilities(100.0, 100.0, 10.0),
        lambda g: g.get_adjacent_cells(g.get_actnum(), 0, 1),
        lambda g: g.get_geometrics(),
        lambda g: g.estimate_design(),
        lambda g: g.ijk_handedness,
        lambda g: setattr(g, "ijk_handedness", "left"),
        lambda g: g.estimate_flip(),
        lambda g: g.reverse_row_axis(ijk_handedness="left"),
        lambda g: g.reverse_column_axis(),
        lambda g: g.reverse_column_axis(ijk_handedness="left"),
        lambda g: xtgeo.surface_from_grid3d(g, where="base"),
        lambda g: xtgeo.surface_from_grid3d(g, where="top"),
        lambda g: xtgeo.surface_from_grid3d(g, where=1),
        lambda g: xtgeo.surface_from_grid3d(g, where=g.nlay, property="raw"),
        lambda g: xtgeo.surface_from_grid3d(g, where=g.nlay, property="i"),
        lambda g: xtgeo.surface_from_grid3d(g, where="1_top"),
        lambda g: xtgeo.surface_from_grid3d(g, where=f"{g.nlay}_base"),
        lambda g: xtgeo.surface_from_grid3d(g, where=2),
        lambda g: xtgeo.surface_from_grid3d(g, where="2_base", property="i"),
        lambda g: g.get_transmissibilities(100.0, 100.0, 10.0),
    ],
    ids=[
        "transmissibilities",
        "adjacent_cells",
        "geometrics",
        "estimate_design",
        "handedness_get",
        "handedness_set",
        "estimate_flip",
        "reverse_row_axis_handedness",
        "reverse_column_axis",
        "reverse_column_axis_handedness",
        "surface_base",
        "surface_top",
        "surface_layer1",
        "surface_last_raw",
        "surface_last_i",
        "surface_1_top",
        "surface_last_base",
        "surface_interior_layer",
        "surface_interior_base_i",
        "deprecated_transmissibilities",
    ],
)
def test_unsupported_operations_raise(nested_box, call):
    with pytest.raises(NotImplementedError, match="nested hybrid"):
        call(nested_box[1])


def test_internal_users_of_geometrics_still_work(nested_box):
    merged = nested_box[1]
    merged.describe(details=True, flush=False)


def test_checking_does_not_convert_xtgformat(nested_box):
    coarse = nested_box[0]
    coarse._set_xtgformat1()
    assert not coarse.nestedhybrid
    assert coarse._xtgformat == 1


def test_conform_to_surfaces_raises(nested_box):
    surf = xtgeo.RegularSurface(
        ncol=5, nrow=5, xinc=100, yinc=100, xori=0, yori=0, values=1510
    )
    with pytest.raises(NotImplementedError, match="nested hybrid"):
        nested_box[1].conform_to_surfaces([surf, surf + 40], [2])


def test_supported_operations_on_nested_grid(nested_box):
    coarse, merged = nested_box
    # Collapsing inactive cells is needed after a merge, so it must not raise.
    merged.collapse_inactive_cells()
    merged.collapse_inactive_cells(internal=False)
    coarse.get_cell_transmissibilities(100.0, 100.0, 10.0)
    coarse.get_geometrics()
    assert coarse.ijk_handedness in ("left", "right")


def _refine_region(grid, region, refinement, names=("P_I", "P_J", "P_K")):
    """Refine the cells in ``region`` the way fmu-tools does, and merge.

    The parent indices are recorded as properties before cropping, so they give an
    independent reference for the mapping.

    Returns:
        The merged grid, the slice of the new refined block, and the recorded parent
        I, J and K of its cells in merged-grid indices, -1 where none.
    """
    grid = grid.copy()
    for prop in grid.get_ijk(names=names, zerobased=True):
        grid.append_prop(prop)
    cells = np.argwhere(region)
    (i0, j0, k0), (i1, j1, k1) = (
        cells.min(axis=0).tolist(),
        (cells.max(axis=0) + 1).tolist(),
    )
    fine = grid.copy()
    fine.crop((i0 + 1, i1), (j0 + 1, j1), (k0 + 1, k1), props="all")
    fine.refine(*refinement)

    # Deactivation masks the parent index properties.
    recorded = [fine.get_prop_by_name(name).values.filled(0) for name in names]
    _deactivate(fine, ~region[tuple(recorded)])
    _deactivate(grid, region)

    layers = np.arange(grid.nlay)
    layer_map = layers + (refinement[2] - 1) * np.clip(layers - k0, 0, k1 - k0)
    merged = xtgeo.grid_merge(
        grid,
        fine,
        layer_map.astype(np.int32),
        (np.arange(fine.nlay) + k0).astype(np.int32),
    )
    block = slice(grid.ncol + 1, merged.ncol)
    expected = np.full((3, fine.ncol, merged.nrow, merged.nlay), -1)
    expected[:, :, : fine.nrow, k0 : k0 + fine.nlay] = (
        recorded[0],
        recorded[1],
        layer_map[recorded[2]],
    )
    return merged, block, tuple(expected)


def _deactivate(grid, cells):
    actnum = grid.get_actnum()
    actnum.values[cells] = 0
    grid.set_actnum(actnum)


def _box(shape, cells):
    region = np.zeros(shape, dtype=bool)
    region[cells] = True
    return region


def _assert_parents(parents, block, recorded, cells):
    """Check parent I, J and K of the given cells of a block against the record."""
    for found, expected in zip((parents.i, parents.j, parents.k), recorded):
        np.testing.assert_array_equal(found[block][cells], expected[cells])


@pytest.mark.parametrize(
    "rotation, refinement, inclined",
    [
        (0, (2, 2, 2), False),
        (30, (3, 2, 1), False),
        (10, (2, 3, 3), True),
        (0, ({1: 2, 2: 3}, 2, 1), False),
    ],
)
def test_parents_match_recorded_parents(rotation, refinement, inclined):
    """Check the mapping against parent indices recorded before refinement."""
    coarse = xtgeo.create_box_grid(
        (8, 7, 5), origin=(1000, 2000, 1500), increment=(100, 80, 10), rotation=rotation
    )
    if inclined:
        coarse._set_xtgformat2()
        coarse._coordsv[:, :, 3] += 30
        noise = np.random.default_rng(1).uniform(-2, 2, coarse._zcornsv.shape[:3])
        coarse._zcornsv = np.maximum.accumulate(
            coarse._zcornsv + noise[..., None], axis=2
        ).astype(np.float32)
    region = _box(coarse.dimensions, np.s_[2:6, 1:5, 1:4])
    grid, block, recorded = _refine_region(coarse, region, refinement)

    parents = grid._get_nestedhybrid_parents()
    active = grid._actnumsv[block] > 0
    _assert_parents(parents, block, recorded, active)
    expected = np.zeros(grid.dimensions, dtype=bool)
    expected[tuple(axis[active] for axis in recorded)] = True
    replaced = parents.celltype == NestedHybridCellType.REPLACED_PARENT
    np.testing.assert_array_equal(replaced, expected)


def test_parents_of_inactive_refined_cells():
    """Refined cells outside an L-shaped region are inactive, in active parents."""
    coarse = xtgeo.create_box_grid((6, 6, 2), increment=(100, 100, 10))
    region = _box(coarse.dimensions, np.s_[1:4, 1:4, :])
    region[3, 3, :] = False
    grid, block, recorded = _refine_region(coarse, region, (2, 2, 1))

    parents = grid._get_nestedhybrid_parents()
    inactive = (grid._actnumsv[block] == 0) & (recorded[0] >= 0)
    assert inactive.any()
    _assert_parents(parents, block, recorded, inactive)
    assert parents.celltype[3, 3, 0] == NestedHybridCellType.ACTIVE_COARSE


def test_parents_in_coarse_columns_before_the_first_active_column():
    """Refining all active cells of the first coarse columns leaves those columns
    without active cells, but they still hold the parents."""
    coarse = xtgeo.create_box_grid((6, 6, 2), increment=(100, 100, 10))
    _deactivate(coarse, np.s_[:1])
    region = _box(coarse.dimensions, np.s_[1:3, :, :])
    grid, block, recorded = _refine_region(coarse, region, (2, 2, 1))
    assert grid.nestedhybrid_blocks[0].istart == 3

    parents = grid._get_nestedhybrid_parents()
    active = grid._actnumsv[block] > 0
    _assert_parents(parents, block, recorded, active)
    expected = np.zeros(grid.dimensions, dtype=bool)
    expected[tuple(axis[active] for axis in recorded)] = True
    replaced = parents.celltype == NestedHybridCellType.REPLACED_PARENT
    np.testing.assert_array_equal(replaced, expected)


@pytest.mark.parametrize("history", ["refined_again", "sibling"])
def test_parents_of_later_block_are_coarse_cells(history):
    """Map a later block to the coarse block, also when it refines a refined block.

    The later block refines the centre of the first refinement again, or it
    refines the coarse cells there directly. The coarse parents are the same.
    """
    coarse = xtgeo.create_box_grid(
        (6, 6, 2), origin=(1000, 2000, 1500), increment=(100, 80, 10), rotation=10
    )
    coarse._set_xtgformat2()
    coarse._coordsv[:, :, 3] += 30
    noise = np.random.default_rng(1).uniform(-2, 2, coarse._zcornsv.shape[:3])
    coarse._zcornsv = np.maximum.accumulate(
        coarse._zcornsv + noise[..., None], axis=2
    ).astype(np.float32)
    centre = np.s_[2:4, 2:4, :]
    region = _box(coarse.dimensions, np.s_[1:5, 1:5, :])
    if history == "sibling":
        region[centre] = False
    first, first_block, first_recorded = _refine_region(coarse, region, (2, 2, 1))
    names = ("Q_I", "Q_J", "Q_K")
    if history == "sibling":
        grid, block, recorded = _refine_region(
            first, _box(first.dimensions, centre), (4, 4, 1), names=names
        )
        active = grid._actnumsv[block] > 0
        expected = tuple(axis[active] for axis in recorded)
    else:
        start = first_block.start
        again = _box(first.dimensions, np.s_[start + 2 : start + 6, 2:6, :])
        grid, block, recorded = _refine_region(first, again, (2, 2, 1), names=names)
        active = grid._actnumsv[block] > 0
        # The recorded parents are first-level cells, which have coarse parents.
        first_level = (recorded[0][active] - start, recorded[1][active])
        expected = tuple(
            axis[(*first_level, recorded[2][active])] for axis in first_recorded
        )

    parents = grid._get_nestedhybrid_parents()
    for found, wanted in zip((parents.i, parents.j, parents.k), expected):
        np.testing.assert_array_equal(found[block][active], wanted)
    first_active = grid._actnumsv[first_block] > 0
    _assert_parents(parents, first_block, first_recorded, first_active)
    assert np.all(parents.celltype[centre] == NestedHybridCellType.REPLACED_PARENT)
    assert np.all(
        parents.celltype[first_block][~first_active]
        == NestedHybridCellType.OTHER_INACTIVE
    )


def test_later_block_ending_inside_a_coarse_cell_has_no_parent_there():
    """Leave the cells unmapped where a block covers only part of a coarse cell."""
    coarse = xtgeo.create_box_grid((6, 6, 2), increment=(100, 100, 10))
    first, first_block, _ = _refine_region(
        coarse, _box(coarse.dimensions, np.s_[1:5, 1:5, :]), (2, 2, 1)
    )
    start = first_block.start
    # The whole of coarse cell (2, 2), and half of its neighbours in I and J.
    region = _box(first.dimensions, np.s_[start + 2 : start + 5, 2:5, :])
    grid, block, recorded = _refine_region(
        first, region, (2, 2, 1), names=("Q_I", "Q_J", "Q_K")
    )

    parents = grid._get_nestedhybrid_parents()
    active = grid._actnumsv[block] > 0
    whole = active & (recorded[0] < start + 4) & (recorded[1] < 4)
    assert whole.any()
    assert np.any(active & ~whole)
    assert np.all(parents.i[block][whole] == 2)
    assert np.all(parents.j[block][whole] == 2)
    np.testing.assert_array_equal(parents.k[block][whole], recorded[2][whole])
    for found in (parents.i, parents.j, parents.k):
        assert np.all(found[block][active & ~whole] == -1)


@pytest.mark.parametrize("squeezed", ["parents", "refined"])
def test_unmatched_depths_give_columns_only(squeezed):
    """Parents that lost their shape, or refined cells that do not fill them, give
    parent columns but no parent layers."""
    coarse = xtgeo.create_box_grid((4, 4, 3), increment=(100, 100, 10))
    region = _box(coarse.dimensions, np.s_[1:3, 1:3, :])
    grid, block, recorded = _refine_region(coarse, region, (2, 2, 1))
    grid._set_xtgformat2()
    if squeezed == "parents":
        # Keep the neighboring cells' corner depths unchanged.
        for di, dj, quadrant in [(0, 0, 3), (1, 0, 2), (1, 1, 0), (0, 1, 1)]:
            corners = grid._zcornsv[1 + di : 3 + di, 1 + dj : 3 + dj, :, quadrant]
            corners[:] = corners[:, :, :1] + (corners - corners[:, :, :1]) / 2
    else:
        # The top refined layer stays inside its parent but no longer fills it.
        corners = grid._zcornsv[block.start : block.stop + 1]
        corners[:] = corners[:, :, :1] + (corners - corners[:, :, :1]) * 0.75

    parents = grid._get_nestedhybrid_parents()
    active = grid._actnumsv[block] > 0
    np.testing.assert_array_equal(parents.i[block][active], recorded[0][active])
    np.testing.assert_array_equal(parents.j[block][active], recorded[1][active])
    assert np.all(parents.k[block] == -1)
    assert np.all(parents.celltype[1:3, 1:3] == NestedHybridCellType.UNVERIFIED_PARENT)
    assert not np.any(parents.celltype == NestedHybridCellType.REPLACED_PARENT)


def test_thin_parent_layers_are_unverified():
    """Classify thin parents as unverified when their K matches are uncertain."""
    coarse = xtgeo.create_box_grid(
        (4, 4, 3), origin=(0, 0, 1500), increment=(100, 100, 0.001)
    )
    region = _box(coarse.dimensions, np.s_[1:3, 1:3, 1:2])
    grid, block, recorded = _refine_region(coarse, region, (2, 2, 2))

    parents = grid._get_nestedhybrid_parents()
    active = grid._actnumsv[block] > 0
    np.testing.assert_array_equal(parents.i[block][active], recorded[0][active])
    np.testing.assert_array_equal(parents.j[block][active], recorded[1][active])
    assert np.all(parents.k[block][active] == -1)
    assert np.all(
        parents.celltype[1:3, 1:3, 1] == NestedHybridCellType.UNVERIFIED_PARENT
    )
    assert np.all(parents.celltype[1:3, 1:3, 2] == NestedHybridCellType.OTHER_INACTIVE)


def test_child_corners_above_inclined_parent_face_keep_parents():
    """Refine interpolates depths along pillars, so on non-parallel inclined pillars
    a refined corner can lie above the parent face. The parent is still known."""
    coarse = xtgeo.create_box_grid(
        (4, 4, 1), origin=(0, 0, 1500), increment=(100, 100, 10)
    )
    coarse._set_xtgformat2()
    coarse._coordsv[..., [2, 5]] = [1480, 1540]
    coarse._coordsv[2, :, [2, 5]] += 10
    coarse._coordsv[2, :, 3] -= 50
    coarse._zcornsv[2] += 20
    region = _box(coarse.dimensions, np.s_[1:2, 1:3, :])
    grid, block, recorded = _refine_region(coarse, region, (2, 2, 1))

    # On the south edge of the parent top, the coarse corners are (100, 1500)
    # and (175, 1520) in X and Z. The midpoint pillar puts the fine corner
    # at X > 137.5 at Z=1510, above the sloping top face.
    middle_pillar = grid._coordsv[block.start + 1, 0]
    fine_top = grid._zcornsv[block.start + 1, 0, 0, 3]
    assert fine_top == 1510
    assert _grid_nested_hybrid._xy_at_depth(middle_pillar, fine_top)[0] > 137.5
    parents = grid._get_nestedhybrid_parents()
    active = grid._actnumsv[block] > 0
    _assert_parents(parents, block, recorded, active)
    replaced = parents.celltype == NestedHybridCellType.REPLACED_PARENT
    assert np.count_nonzero(replaced) == np.count_nonzero(region)


def test_parent_mapping_is_independent_of_chunk_size(monkeypatch):
    """Keep the same mapping when vertical and inclined columns share a chunk."""
    coarse = xtgeo.create_box_grid((5, 4, 1), increment=(100, 100, 10))
    coarse._set_xtgformat2()
    coarse._coordsv[4, :, 3] += 5
    region = _box(coarse.dimensions, np.s_[1:4, 1:3, :])
    grid, block, recorded = _refine_region(coarse, region, (2, 2, 2))
    # Offset one vertical pillar within the tolerance for matching columns.
    grid._coordsv[block.start + 1, 1, [0, 3]] += 0.002

    mixed_chunk = grid._get_nestedhybrid_parents()
    monkeypatch.setattr(_grid_nested_hybrid, "_CHUNK_SIZE", 1)
    separate_chunks = grid.copy()._get_nestedhybrid_parents()

    active = grid._actnumsv[block] > 0
    _assert_parents(separate_chunks, block, recorded, active)
    _assert_parents(mixed_chunk, block, recorded, active)
    np.testing.assert_array_equal(mixed_chunk.celltype, separate_chunks.celltype)


@pytest.mark.parametrize(
    "first_shifted_pillar", [5, 6], ids=["no_matching_line", "one_matching_line"]
)
def test_moved_refined_block_has_no_parents(nested_box, first_shifted_pillar):
    """Leave the block unmapped when too few pillar lines still match."""
    grid = nested_box[1]
    grid._set_xtgformat2()
    grid._coordsv[first_shifted_pillar:, :, [0, 3]] += 5
    assert grid.nestedhybrid

    parents = grid._get_nestedhybrid_parents()
    assert np.all(parents.i == -1)
    assert np.all(parents.k == -1)
    assert not np.isin(
        parents.celltype,
        [NestedHybridCellType.REPLACED_PARENT, NestedHybridCellType.UNVERIFIED_PARENT],
    ).any()


def test_rotated_refined_block_with_conflicting_pillar_lines_has_no_parents():
    """Leave a rotated block unmapped when its pillar lines conflict."""
    coarse = xtgeo.create_box_grid((4, 4, 1), increment=(100, 100, 10))
    refined = xtgeo.create_box_grid(
        (4, 4, 1), origin=(200, 100, 0), increment=(50, 50, 10), rotation=90
    )
    grid = xtgeo.grid_merge(coarse, refined)

    parents = grid._get_nestedhybrid_parents()
    assert parents is not None
    assert np.all(parents.i == -1)
    assert np.all(parents.j == -1)


def test_later_block_without_horizontal_area_has_no_parents(nested_box):
    """Leave a block with no horizontal area unmapped and keep earlier mappings."""
    later = xtgeo.create_box_grid((2, 2, 1), increment=(100, 100, 10))
    later._set_xtgformat2()
    later._coordsv[..., [0, 3]] = 0
    grid = xtgeo.grid_merge(nested_box[1], later)
    blocks = grid.nestedhybrid_blocks
    assert len(blocks) == 3

    parents = grid._get_nestedhybrid_parents()
    assert np.all(parents.i[blocks[2].istart :] == -1)
    assert np.any(parents.k[blocks[1].istart : blocks[1].istop] >= 0)


def test_celltypes(nested_box):
    """Distinguish replaced parents from padding and check the padding layers."""
    grid = nested_box[1]
    active = grid._actnumsv > 0
    expected = np.where(
        active, NestedHybridCellType.ACTIVE_REFINED, NestedHybridCellType.OTHER_INACTIVE
    )
    expected[:4][active[:4]] = NestedHybridCellType.ACTIVE_COARSE
    expected[1:3, 1:3, 1] = NestedHybridCellType.REPLACED_PARENT

    parents = grid._get_nestedhybrid_parents()
    np.testing.assert_array_equal(parents.celltype, expected)
    # Layers 0 and 3 are merge padding.
    assert np.all(parents.k[5:, :, [0, 3]] == -1)
    assert np.all(parents.k[5:, :, 1:3] == 1)


def test_inactive_cell_above_proven_children_is_not_a_parent(nested_box):
    """All refined cells in the column have a proven parent layer, so the other
    inactive cells in the column are not parents."""
    grid = nested_box[1]
    _deactivate(grid, (1, 1, 0))

    celltype = grid._get_nestedhybrid_parents().celltype
    assert celltype[1, 1, 0] == NestedHybridCellType.OTHER_INACTIVE
    assert celltype[1, 1, 1] == NestedHybridCellType.REPLACED_PARENT


def test_parents_none_when_not_nested():
    """Return None for a grid with no nested hybrid layout."""
    assert xtgeo.create_box_grid((3, 3, 1))._get_nestedhybrid_parents() is None


def test_parents_are_cached(nested_box, monkeypatch):
    """Reuse the computed parent mapping when the grid stays the same."""
    grid = nested_box[1]
    map_parents = _grid_nested_hybrid.map_parents
    calls = 0

    def count_calls(*args):
        nonlocal calls
        calls += 1
        return map_parents(*args)

    monkeypatch.setattr(_grid_nested_hybrid, "map_parents", count_calls)
    first = grid._get_nestedhybrid_parents()
    assert grid._get_nestedhybrid_parents() is first
    assert calls == 1


def test_parents_follow_grid_edits(nested_box):
    """Preserve the mapping on copies and update it after grid edits."""
    grid = nested_box[1]
    before = grid._get_nestedhybrid_parents()
    copied = grid.copy()._get_nestedhybrid_parents()
    for name in ("i", "j", "k", "celltype"):
        np.testing.assert_array_equal(getattr(copied, name), getattr(before, name))

    actnum = grid.get_actnum()
    actnum.values[1, 1, 1] = 1
    grid.set_actnum(actnum)
    assert (
        grid._get_nestedhybrid_parents().celltype[1, 1, 1]
        == NestedHybridCellType.ACTIVE_COARSE
    )

    grid._set_xtgformat2()
    grid._coordsv[6, 2, [0, 3]] += 1
    assert grid._get_nestedhybrid_parents().i[6, 2, 1] == -1


def test_parents_drogon(testdata_path):
    """Every active refined cell has its centre inside its parent column."""
    grid = xtgeo.grid_from_file(
        pathlib.Path(testdata_path) / "3dgrids/drogon/5/drogon_nested_hybrid1.roff"
    )
    parents = grid._get_nestedhybrid_parents()
    block = grid.nestedhybrid_blocks[1]
    active = np.zeros(grid.dimensions, dtype=bool)
    active[block.istart : block.istop] = grid._actnumsv[block.istart : block.istop] > 0
    pi, pj = parents.i[active], parents.j[active]
    assert np.all(pi >= 0)

    x, y, z = (prop.values.data[active] for prop in grid.get_xyz(asmasked=False))
    pillars = [
        grid._coordsv[pi + di, pj + dj] for di, dj in [(0, 0), (1, 0), (1, 1), (0, 1)]
    ]
    corners = np.stack(
        [_grid_nested_hybrid._xy_at_depth(pillar, z) for pillar in pillars], axis=1
    )
    assert shapely.contains_xy(shapely.polygons(corners), x, y).all()
    replaced = parents.celltype == NestedHybridCellType.REPLACED_PARENT
    assert np.all(grid._actnumsv[replaced] == 0)


@pytest.fixture
def emerald_nested_grid(testdata_path):
    return xtgeo.grid_from_file(
        pathlib.Path(testdata_path)
        / "3dgrids/eme/3/nested_grid_refine_2_2_2_merged.grdecl"
    )


def test_parents_emerald(emerald_nested_grid):
    """Every refined cell of the 2 x 2 x 2 refinement has a parent in a group of 8."""
    grid = emerald_nested_grid
    parents = grid._get_nestedhybrid_parents()
    refined = parents.celltype == NestedHybridCellType.ACTIVE_REFINED
    for axis in (parents.i, parents.j, parents.k):
        assert np.all(axis[refined] >= 0)
    index = (parents.i[refined], parents.j[refined], parents.k[refined])
    _, counts = np.unique(np.column_stack(index), axis=0, return_counts=True)
    assert np.all(counts == 8)
    assert np.all(parents.celltype[index] == NestedHybridCellType.REPLACED_PARENT)
    assert np.count_nonzero(
        parents.celltype == NestedHybridCellType.REPLACED_PARENT
    ) == len(counts)


@pytest.mark.benchmark(group="nested-hybrid-parent-mapping")
def test_benchmark_parent_mapping_emerald(emerald_nested_grid, benchmark):
    """Time parent mapping after grid loading and block detection."""
    grid = emerald_nested_grid
    blocks = grid.nestedhybrid_blocks
    parents = benchmark(_grid_nested_hybrid.map_parents, grid, blocks)
    assert np.any(parents.celltype == NestedHybridCellType.REPLACED_PARENT)


def test_inverted_parent_depths_give_no_parent_layer():
    """Small depth inversions add up, so two parent layers contain the same cells."""
    coarse = xtgeo.create_box_grid((4, 4, 11), increment=(100, 100, 1))
    coarse._set_xtgformat2()
    coarse._zcornsv[:] = (1500 + np.r_[-0.004, np.zeros(10), 0.004])[:, None]
    _deactivate(coarse, np.s_[1:3, 1:3, :10])
    region = _box(coarse.dimensions, np.s_[1:3, 1:3, 10:])
    grid, block, _ = _refine_region(coarse, region, (2, 2, 1))
    grid._set_xtgformat2()
    inverted = np.r_[0, 0.004, 0.0035, 0.003, 0.0025, 0.002, 0.0015, 0.001]
    grid._zcornsv[1:4, 1:4] = (1500 + np.r_[inverted, 0.0005, 0, 0, 0.004])[:, None]

    parents = grid._get_nestedhybrid_parents()
    assert np.all(parents.k[block] == -1)
    assert not np.any(parents.celltype == NestedHybridCellType.REPLACED_PARENT)


def test_inverted_child_depths_leave_affected_parent_unverified():
    """Reject an inverted child column and keep the neighboring parent verified."""
    coarse = xtgeo.create_box_grid(
        (4, 4, 1), origin=(0, 0, 1500), increment=(100, 100, 10)
    )
    region = _box(coarse.dimensions, np.s_[1:3, 1:2, :])
    grid, block, recorded = _refine_region(coarse, region, (2, 2, 3))
    depths = np.array([1500, 1503, 1507, 1510])
    grid._zcornsv[block.start : block.stop + 1] = depths[:, None]
    grid._zcornsv[block.start, 0, 1:3, 3] = [1507, 1503]

    parents = grid._get_nestedhybrid_parents()
    expected = (recorded[0], recorded[1], np.where(recorded[0] == 1, -1, recorded[2]))
    _assert_parents(parents, block, expected, grid._actnumsv[block] > 0)
    assert parents.celltype[1, 1, 0] == NestedHybridCellType.UNVERIFIED_PARENT
    assert parents.celltype[2, 1, 0] == NestedHybridCellType.REPLACED_PARENT


def test_garbage_depths_outside_the_blocks_do_not_change_parents():
    """Keep verified parent layers when a separator contains an invalid depth."""
    coarse = xtgeo.create_box_grid((4, 4, 3), increment=(100, 100, 10))
    region = _box(coarse.dimensions, np.s_[1:3, 1:3, :])
    grid, block, recorded = _refine_region(coarse, region, (2, 2, 1))
    grid._set_xtgformat2()
    grid._zcornsv[4, 0, 0, 3] = 1e33  # in the inactive separator column

    parents = grid._get_nestedhybrid_parents()
    active = grid._actnumsv[block] > 0
    np.testing.assert_array_equal(parents.k[block][active], recorded[2][active])


def test_parent_needs_all_its_refined_columns(monkeypatch):
    """Reject a partially filled parent when processing columns in small chunks."""
    coarse = xtgeo.create_box_grid((6, 6, 2), increment=(100, 100, 10))
    region = _box(coarse.dimensions, np.s_[1:5, 1:5, :])
    grid, block, recorded = _refine_region(coarse, region, (2, 2, 2))
    monkeypatch.setattr(_grid_nested_hybrid, "_CHUNK_SIZE", 1)
    grid._set_xtgformat2()
    column = (block.start + 2, 2)
    for di, dj, quadrant in [(0, 0, 3), (1, 0, 2), (1, 1, 0), (0, 1, 1)]:
        corners = grid._zcornsv[column[0] + di, column[1] + dj, :, quadrant]
        corners[:] = corners[0] + (corners - corners[0]) * 0.75

    parents = grid._get_nestedhybrid_parents()
    local = (column[0] - block.start, column[1])
    in_parent_column = (recorded[0] == recorded[0][local][0]) & (
        recorded[1] == recorded[1][local][0]
    )
    active = grid._actnumsv[block] > 0
    expected = np.where(in_parent_column, -1, recorded[2])
    np.testing.assert_array_equal(parents.k[block][active], expected[active])


def test_moved_interior_pillar_gives_no_parent_layer():
    """Refined columns on a moved pillar lose their parent, so the remaining columns
    no longer prove that the parent is filled."""
    coarse = xtgeo.create_box_grid((4, 4, 2), increment=(90, 90, 10))
    region = _box(coarse.dimensions, np.s_[1:2, 1:2, :])
    grid, block, recorded = _refine_region(coarse, region, (3, 3, 1))
    grid._set_xtgformat2()
    grid._coordsv[block.start + 1, 1, [0, 3]] += 5

    parents = grid._get_nestedhybrid_parents()
    mapped = parents.i[block] >= 0
    assert 0 < mapped[..., 0].sum() < 9
    np.testing.assert_array_equal(parents.i[block][mapped], recorded[0][mapped])
    assert np.all(parents.k == -1)
    assert np.all(parents.celltype[1, 1] == NestedHybridCellType.UNVERIFIED_PARENT)


def test_padding_inside_parent_keeps_parent_layers():
    """Refining one first-level layer again adds zero-thickness padding between the
    first-level layers that fill a coarse parent."""
    coarse = xtgeo.create_box_grid((6, 6, 2), increment=(100, 100, 10))
    first, first_block, _ = _refine_region(
        coarse, _box(coarse.dimensions, np.s_[1:5, 1:5, :]), (2, 2, 2)
    )
    start = first_block.start
    region = _box(first.dimensions, np.s_[start + 2 : start + 4, 2:4, 0:1])
    grid, _, _ = _refine_region(first, region, (2, 2, 2), names=("Q_I", "Q_J", "Q_K"))
    assert np.any(np.diff(grid._zcornsv[first_block.start, 0, :, 3]) == 0)

    parents = grid._get_nestedhybrid_parents()
    active = grid._actnumsv[first_block] > 0
    assert np.all(parents.k[first_block][active] >= 0)


def test_positive_thickness_between_child_layers_is_not_padding():
    """Reject a parent match when an unproven child layer has positive thickness."""
    coarse = xtgeo.create_box_grid(
        (4, 4, 1), origin=(0, 0, 1500), increment=(100, 100, 10)
    )
    region = _box(coarse.dimensions, np.s_[1:2, 1:3, :])
    grid, block, _ = _refine_region(coarse, region, (2, 2, 3))
    next_depth = np.nextafter(np.float32(1505), np.float32(np.inf))
    grid._zcornsv[block.start : block.stop + 1] = np.array(
        [1500, 1505, next_depth, 1510]
    )[:, None]

    parents = grid._get_nestedhybrid_parents()
    assert np.all(parents.k[block] == -1)
    assert not np.any(parents.celltype == NestedHybridCellType.REPLACED_PARENT)
