import pathlib

import numpy as np
import pytest

import xtgeo
from xtgeo.common.log import null_logger

logger = null_logger(__name__)

RPATH1 = pathlib.Path("surfaces/reek")
RPATH2 = pathlib.Path("3dgrids/reek")

RTOP1 = RPATH1 / "1/topreek_rota.gri"
RGRD1 = RPATH2 / "REEK.EGRID"
RPROP1 = RPATH2 / "REEK.INIT"
RGRD2 = RPATH2 / "reek_sim_grid.roff"
RPROP2 = RPATH2 / "reek_sim_zone.roff"

EMERALD_CLIPPED = pathlib.Path("3dgrids/eme/3/original.grdecl")


def test_get_surface_from_grd3d_porosity(tmp_path, generate_plot, testdata_path):
    """Sample a surface from a 3D grid"""

    surf = xtgeo.surface_from_file(testdata_path / RTOP1)
    grd = xtgeo.grid_from_file(testdata_path / RGRD1, fformat="egrid")
    surf.values = 1700
    zsurf = surf.copy()
    surfr = surf.copy()
    surf2 = surf.copy()
    phi = xtgeo.gridproperty_from_file(
        testdata_path / RPROP1, fformat="init", name="PORO", grid=grd
    )

    # slice grd3d
    surf.slice_grid3d(grd, phi)

    surf.to_file(tmp_path / "surf_slice_grd3d_reek.gri")
    if generate_plot:
        surf.quickplot(filename=tmp_path / "surf_slice_grd3d_reek.png")

    # refined version:
    surfr.refine(2)
    surfr.slice_grid3d(grd, phi)

    surfr.to_file(tmp_path / "surf_slice_grd3d_reek_refined.gri")
    if generate_plot:
        surfr.quickplot(filename=tmp_path / "surf_slice_grd3d_reek_refined.png")

    # use zsurf:
    surf2.slice_grid3d(grd, phi, zsurf=zsurf)

    surf2.to_file(tmp_path / "surf_slice_grd3d_reek_zslice.gri")
    if generate_plot:
        surf2.quickplot(filename=tmp_path / "surf_slice_grd3d_reek_zslice.png")

    assert np.allclose(surf.values, surf2.values)

    assert surf.values.mean() == pytest.approx(0.1667, abs=0.01)
    assert surfr.values.mean() == pytest.approx(0.1667, abs=0.01)


def test_get_surface_from_grd3d_zones(tmp_path, generate_plot, testdata_path):
    """Sample a surface from a 3D grid, using zones"""

    surf = xtgeo.surface_from_file(testdata_path / RTOP1)
    grd = xtgeo.grid_from_file(testdata_path / RGRD2, fformat="roff")
    surf.values = 1700
    zone = xtgeo.gridproperty_from_file(
        testdata_path / RPROP2, fformat="roff", name="Zone", grid=grd
    )

    # slice grd3d
    surf.slice_grid3d(grd, zone, sbuffer=1)

    surf.to_file(tmp_path / "surf_slice_grd3d_reek_zone.gri")
    if generate_plot:
        surf.quickplot(filename=tmp_path / "surf_slice_grd3d_reek_zone.png")


@pytest.mark.filterwarnings("ignore:Default values*")
def test_surface_from_grd3d_layer(
    tmp_path, generate_plot, default_surface, testdata_path
):
    """Create a surface from a 3D grid layer"""

    surf = xtgeo.RegularSurface(**default_surface)
    grd = xtgeo.grid_from_file(testdata_path / RGRD2, fformat="roff")
    surf = xtgeo.surface_from_grid3d(grd)

    surf.fill()
    surf.to_file(tmp_path / "surf_from_grid3d_top.gri")
    tmp = surf.copy()
    if generate_plot:
        surf.quickplot(filename=tmp_path / "surf_from_grid3d_top.png")

    surf = xtgeo.surface_from_grid3d(grd, template=tmp, property="i")

    surf.to_file(tmp_path / "surf_from_grid3d_top_icell.gri")
    if generate_plot:
        surf.quickplot(filename=tmp_path / "surf_from_grid3d_top_icell.png")

    surf = xtgeo.surface_from_grid3d(grd, template=tmp, property="j")
    surf.fill()
    surf.to_file(tmp_path / "surf_from_grid3d_top_jcell.gri")
    if generate_plot:
        surf.quickplot(filename=tmp_path / "surf_from_grid3d_top_jcell.png")

    surf = xtgeo.surface_from_grid3d(
        grd, template=tmp, property="depth", where="3_base"
    )
    assert surf.values.mean() == pytest.approx(1714.304, abs=0.01)
    if generate_plot:
        surf.quickplot(filename=tmp_path / "surf_from_grid3d_3base.png")


def test_surface_from_grd3d_layer_native(testdata_path):
    """Create a surface from a 3D grid layer, using 'native' map resolution."""

    grd = xtgeo.grid_from_file(testdata_path / RGRD2, fformat="roff")

    template = "native"
    depth = xtgeo.surface_from_grid3d(grd, template=template, property="depth")

    assert depth.ncol == grd.ncol
    assert depth.nrow == grd.nrow
    assert depth.values.mean() == pytest.approx(1701.074, abs=0.01)


def test_surface_from_grd3d_layer_raw(testdata_path):
    """Get multiple returns from a 3D grid layer, for own processing"""

    grd = xtgeo.grid_from_file(testdata_path / RGRD2, fformat="roff")

    template = "native"
    ii, _, top, _, _ = xtgeo.surface_from_grid3d(
        grd, template=template, property="raw", where=3
    )
    assert ii.shape == (grd.ncol, grd.nrow)

    assert np.nanmean(top) == pytest.approx(1709.754, abs=0.01)


@pytest.mark.parametrize("where", [0, -1, 3])
def test_surface_from_grid3d_rejects_invalid_layer(where):
    """Reject layer numbers outside 1..nlay."""
    grid = xtgeo.create_box_grid((2, 2, 2))
    with pytest.raises(ValueError, match="Klayer out of range"):
        xtgeo.surface_from_grid3d(grid, template="native", where=where)


def test_raw_boundary_properties_use_per_column_layers():
    """Recover K from activity when raw base samples use different stored layers."""
    coarse = xtgeo.create_box_grid((2, 2, 2), increment=(1, 1, 10))
    other = xtgeo.create_box_grid((2, 2, 1), origin=(10, 0, 0), increment=(1, 1, 20))
    grid = xtgeo.grid_merge(coarse, other, np.array([0, 1]), np.array([0]))
    template = xtgeo.RegularSurface(
        ncol=13, nrow=2, xori=0.5, yori=0.5, xinc=1, yinc=1, values=0
    )
    prop = xtgeo.GridProperty(grid, values=np.indices(grid.dimensions)[2])
    ii, jj, _, _, inactive = xtgeo.surface_from_grid3d(
        grid, template=template, where="base", property="raw"
    )
    present = grid.get_actnum().values > 0
    layers = grid.nlay - 1 - present[:, :, ::-1].argmax(axis=2)
    valid = (ii >= 0) & ~inactive
    kk = layers[ii[valid], jj[valid]]
    expected = xtgeo.surface_from_grid3d(
        grid, template=template, where="base", property=prop
    )
    assert set(kk) == {0, 1}
    np.testing.assert_array_equal(~valid, expected.values.mask)
    np.testing.assert_array_equal(
        prop.values[ii[valid], jj[valid], kk], expected.values[valid]
    )


def test_surface_from_grd3d_layer_deprecate_mode(testdata_path):
    """Create a surface from a 3D grid layer"""

    grd = xtgeo.grid_from_file(testdata_path / RGRD2, fformat="roff")

    with pytest.warns(FutureWarning):
        _ = xtgeo.surface_from_grid3d(grd, where="3_base", mode="depth")


@pytest.fixture
def nested_box_with_template(nested_box):
    """Add a fixed surface template to the shared coarse and nested boxes.

    The template has 64 sample points inside the original box.
    """
    coarse, merged = nested_box
    template = xtgeo.RegularSurface(
        ncol=8, nrow=8, xori=25, yori=25, xinc=50, yinc=50, values=0
    )
    return coarse, merged, template


@pytest.mark.parametrize("where, depth", [("top", 1500), ("base", 1560)])
def test_nested_box_boundary(nested_box_with_template, where, depth):
    """Keep the top at 1500 m and base at 1560 m after middle-interval refinement."""
    coarse, merged, template = nested_box_with_template
    original = xtgeo.surface_from_grid3d(coarse, template=template, where=where)
    result = xtgeo.surface_from_grid3d(merged, template=template, where=where)
    np.testing.assert_array_equal(result.values.mask, original.values.mask)
    np.testing.assert_allclose(result.values, depth)


@pytest.mark.parametrize("where", ["top", "base"])
def test_boundary_excludes_inactive_cells(nested_box_with_template, where):
    """Mask the boundary when all cells are inactive."""
    coarse, merged, template = nested_box_with_template
    for grid in (coarse, merged):
        actnum = grid.get_actnum()
        actnum.values[:] = 0
        grid.set_actnum(actnum)
        result = xtgeo.surface_from_grid3d(grid, template=template, where=where)
        assert result.values.count() == 0


def test_nested_box_explicit_layer(nested_box_with_template):
    """Sample stored layer 3, including inactive cells in the depth map."""
    _, merged, template = nested_box_with_template
    result = xtgeo.surface_from_grid3d(merged, template=template, where="3_base")
    assert result.values.count() == 64
    np.testing.assert_allclose(result.values.compressed(), 1540)
    prop = xtgeo.GridProperty(merged, values=1540)
    values = xtgeo.surface_from_grid3d(
        merged, template=template, where="3_base", property=prop
    )
    assert values.values.count() == 16
    assert values.values.mask[0, 0]
    assert values.values[3, 3] == 1540


@pytest.mark.parametrize("where", ["top", "base"])
def test_nested_boundary_indices_and_properties(nested_box_with_template, where):
    """Return indices and property values from the selected boundary cells.

    Each value is 100 * I + K, so a wrong column or layer changes the result.
    Sampling must also leave the input property's mask unchanged.
    """
    _, merged, template = nested_box_with_template
    ii, jj = np.indices((4, 4))
    prop_values = (
        np.broadcast_to(np.arange(merged.nlay), merged.dimensions).copy()
        + 100 * np.indices(merged.dimensions)[0]
    )
    prop = xtgeo.GridProperty(merged, values=prop_values)
    original_mask = np.ma.getmaskarray(prop.values).copy()
    i_surface = xtgeo.surface_from_grid3d(
        merged,
        template=template,
        where=where,
        property="i",
        index_position=where,
    )
    j_surface = xtgeo.surface_from_grid3d(
        merged,
        template=template,
        where=where,
        property="j",
        index_position=where,
    )
    result = xtgeo.surface_from_grid3d(
        merged, template=template, where=where, property=prop
    )
    np.testing.assert_array_equal(
        i_surface.values, np.repeat(np.repeat(ii, 2, 0), 2, 1)
    )
    np.testing.assert_array_equal(
        j_surface.values, np.repeat(np.repeat(jj, 2, 0), 2, 1)
    )
    expected = 100 * np.repeat(np.repeat(ii, 2, 0), 2, 1)
    expected += 0 if where == "top" else 3
    np.testing.assert_array_equal(result.values, expected)
    np.testing.assert_array_equal(np.ma.getmaskarray(prop.values), original_mask)


@pytest.mark.parametrize("where", ["top", "base"])
def test_refined_boundary_cell_wins_over_inactive_parent(where):
    """Prefer an active refined cell over an inactive parent at the same depth.

    The property is 10 * I + K. The selected refined cell has I=3 and K=0
    at the top or K=3 at the base, giving values of 30 and 33 respectively.
    """
    coarse = xtgeo.create_box_grid((2, 2, 2), increment=(100, 100, 20))
    fine = coarse.copy()
    fine.crop((1, 1), (1, 1), (1, 2))
    fine.refine(2, 2, 2)
    actnum = coarse.get_actnum()
    actnum.values[0, 0, :] = 0
    coarse.set_actnum(actnum)
    merged = xtgeo.grid_merge(coarse, fine, np.array([0, 2]), np.array([0, 1, 2, 3]))
    template = xtgeo.RegularSurface(
        ncol=4, nrow=4, xori=25, yori=25, xinc=50, yinc=50, values=0
    )
    prop = xtgeo.GridProperty(
        merged,
        values=np.indices(merged.dimensions)[0] * 10 + np.indices(merged.dimensions)[2],
    )
    result = xtgeo.surface_from_grid3d(
        merged, template=template, where=where, property=prop
    )
    assert result.values[0, 0] == (30 if where == "top" else 33)
    assert not result.values.mask[0, 0]


@pytest.fixture
def uneven_nested_template():
    """Return a surface template for the shared uneven nested grid.

    Each of the 64 sample points is at the center of one refined column.
    """
    return xtgeo.RegularSurface(
        ncol=8, nrow=8, xori=125, yori=125, xinc=50, yinc=50, values=0
    )


@pytest.mark.parametrize("geometry", ["tilted", "non-planar"])
@pytest.mark.parametrize("refined_first", [False, True])
@pytest.mark.parametrize("where", ["top", "base", "1_top", "3_base"])
def test_uneven_nested_grid_samples_refined_cells(
    uneven_nested_grid, uneven_nested_template, geometry, refined_first, where
):
    """Return the active refined cells when parent and refined depths differ.

    The property value is the merged-grid I index, so a value from an inactive
    parent changes the result.
    """
    _, _, merged = uneven_nested_grid(geometry, refined_first)
    template = uneven_nested_template
    prop = xtgeo.GridProperty(merged, values=np.indices(merged.dimensions)[0])
    first_refined_column = 0 if refined_first else 7
    expected = np.broadcast_to(first_refined_column + np.arange(8)[:, None], (8, 8))

    values = xtgeo.surface_from_grid3d(
        merged, template=template, where=where, property=prop
    )
    indices = xtgeo.surface_from_grid3d(
        merged, template=template, where=where, property="i"
    )
    *_, inactive = xtgeo.surface_from_grid3d(
        merged, template=template, where=where, property="raw"
    )

    assert values.values.count() == 64
    np.testing.assert_array_equal(values.values, expected)
    np.testing.assert_array_equal(indices.values, expected)
    assert not inactive.any()


@pytest.mark.parametrize("geometry", ["tilted", "non-planar"])
@pytest.mark.parametrize("where", ["top", "base"])
def test_uneven_nested_grid_depth_excludes_inactive_parents(
    uneven_nested_grid, uneven_nested_template, geometry, where
):
    """Use refined face depths when the parent cells are inactive."""
    _, fine, merged = uneven_nested_grid(geometry)
    template = uneven_nested_template
    refined = xtgeo.surface_from_grid3d(fine, template=template, where=where)

    result = xtgeo.surface_from_grid3d(merged, template=template, where=where)

    assert result.values.count() == 64
    np.testing.assert_array_equal(result.values, refined.values)


@pytest.fixture
def emerald_nested_grid(testdata_path):
    """Return the clipped Emerald grid, a nested grid built from it and a template.

    Columns 11 to 20 and rows 21 to 30 are refined by two in each axis through
    all layers, and their parent cells are inactive. The PARENT property holds
    the original column and row of each cell, also in the refined cells.
    """
    original = xtgeo.grid_from_file(testdata_path / EMERALD_CLIPPED)
    columns, rows, _ = np.indices(original.dimensions)
    original.append_prop(
        xtgeo.GridProperty(original, values=1000.0 * columns + rows, name="PARENT")
    )
    coarse = original.copy()
    fine = original.copy()
    fine.crop((11, 20), (21, 30), (1, original.nlay), props="all")
    fine.refine(2, 2, 2)
    actnum = coarse.get_actnum()
    actnum.values[10:20, 20:30, :] = 0
    coarse.set_actnum(actnum)
    merged = xtgeo.grid_merge(
        coarse, fine, 2 * np.arange(original.nlay), np.arange(2 * original.nlay)
    )
    template = xtgeo.surface_from_grid3d(original)
    return original, merged, template


@pytest.mark.parametrize("where", ["top", "base"])
def test_nested_real_grid_matches_original_grid(emerald_nested_grid, where):
    """Give the same surfaces from a nested real grid as from the original grid.

    Refined faces are not exactly on the non-planar parent faces, so depths
    agree within 0.5 m. Property values must be equal at the same map points.
    """
    original, merged, template = emerald_nested_grid
    expected_depth = xtgeo.surface_from_grid3d(
        original, template=template, where=where
    ).values
    expected_parent = xtgeo.surface_from_grid3d(
        original,
        template=template,
        where=where,
        property=original.get_prop_by_name("PARENT"),
        index_position=where,
    ).values

    depth = xtgeo.surface_from_grid3d(merged, template=template, where=where).values
    parent = xtgeo.surface_from_grid3d(
        merged,
        template=template,
        where=where,
        property=merged.get_prop_by_name("PARENT"),
        index_position=where,
    ).values

    np.testing.assert_array_equal(
        np.ma.getmaskarray(depth), np.ma.getmaskarray(expected_depth)
    )
    np.testing.assert_allclose(
        depth.compressed(), expected_depth.compressed(), atol=0.5
    )
    np.testing.assert_array_equal(
        np.ma.getmaskarray(parent), np.ma.getmaskarray(expected_parent)
    )
    np.testing.assert_array_equal(parent.compressed(), expected_parent.compressed())


@pytest.mark.parametrize("where, depth", [("top", 6), ("base", 4)])
@pytest.mark.parametrize("refined_first", [False, True])
def test_boundary_cell_wins_over_center_fallback(where, depth, refined_first):
    """Prefer a real boundary to an inclined cell's center in either merge order.

    Both cells are active. The coarse cell center is at depth 5. Its face misses the
    sample points, so its center must not override the refined boundary.
    """
    coarse = xtgeo.create_box_grid((1, 1, 1), increment=(1, 1, 10))
    coarse._coordsv[:, :, 0 if where == "top" else 3] += 1
    fine = xtgeo.create_box_grid(
        (1, 1, 1),
        origin=(0.6, 0, 6 if where == "top" else 3),
        increment=(0.2, 1, 1),
    )
    merged = (
        xtgeo.grid_merge(fine, coarse)
        if refined_first
        else xtgeo.grid_merge(coarse, fine)
    )
    expected_i = 0 if refined_first else 2
    template = xtgeo.RegularSurface(
        ncol=2, nrow=2, xori=0.65, yori=0.25, xinc=0.05, yinc=0.25, values=0
    )
    prop = xtgeo.GridProperty(merged, values=np.indices(merged.dimensions)[0] + 10)

    depths = xtgeo.surface_from_grid3d(merged, template=template, where=where)
    indices = xtgeo.surface_from_grid3d(
        merged, template=template, where=where, property="i"
    )
    values = xtgeo.surface_from_grid3d(
        merged, template=template, where=where, property=prop
    )

    np.testing.assert_allclose(depths.values, depth)
    np.testing.assert_array_equal(indices.values, expected_i)
    assert values.values.count() == 4
    np.testing.assert_array_equal(values.values, expected_i + 10)


def test_padding_does_not_extend_surface_coverage():
    """Keep points between two separate boxes masked despite the merge separator."""
    merged = xtgeo.grid_merge(
        xtgeo.create_box_grid((1, 1, 1)),
        xtgeo.create_box_grid((1, 1, 1), origin=(10, 0, 0)),
    )
    template = xtgeo.RegularSurface(
        ncol=11, nrow=2, xori=0.5, yori=0.25, xinc=1, yinc=0.5, values=0
    )
    result = xtgeo.surface_from_grid3d(merged, template=template)
    assert result.values.count() == 4
    assert result.values.mask[1:10].all()


@pytest.mark.benchmark(group="surface-from-grid3d")
@pytest.mark.parametrize("case", ["layer", "boundary"])
def test_benchmark_surface_boundary(nested_box, benchmark, case):
    """Time layer and boundary sampling on a map where the per-point cost dominates.

    The grid setup is outside the timer. All 250 000 map points are inside the box.
    """
    coarse, _ = nested_box
    template = xtgeo.RegularSurface(
        ncol=500, nrow=500, xori=0.4, yori=0.4, xinc=0.8, yinc=0.8, values=0
    )
    where = "3_base" if case == "layer" else "base"
    result = benchmark(
        xtgeo.surface_from_grid3d, coarse, template=template, where=where
    )
    assert result.values.count() == 250_000


@pytest.mark.parametrize("where, depth, layer", [("top", 10, 1), ("base", 30, 2)])
@pytest.mark.parametrize("fformat", [None, "roff", "grdecl", "egrid"])
def test_boundary_uses_active_layers(tmp_path, where, depth, layer, fformat):
    """Skip inactive outer layers and mask columns with no active cells."""
    grid = xtgeo.create_box_grid((2, 2, 4), increment=(1, 1, 10))
    actnum = grid.get_actnum()
    actnum.values[:, :, [0, 3]] = 0
    actnum.values[0, 0, :] = 0
    grid.set_actnum(actnum)
    if fformat is not None:
        path = tmp_path / f"grid.{fformat}"
        grid.to_file(path, fformat=fformat)
        grid = xtgeo.grid_from_file(path, fformat=fformat)
    template = xtgeo.RegularSurface(
        ncol=2, nrow=2, xori=0.5, yori=0.5, xinc=1, yinc=1, values=0
    )
    expected_mask = np.array([[True, False], [False, False]])
    prop = xtgeo.GridProperty(grid, values=np.indices(grid.dimensions)[2])
    for property, expected in (("depth", depth), (prop, layer)):
        result = xtgeo.surface_from_grid3d(
            grid, template=template, where=where, property=property
        )
        np.testing.assert_array_equal(result.values.mask, expected_mask)
        np.testing.assert_allclose(result.values.compressed(), expected)
    ii, jj, top, base, inactive = xtgeo.surface_from_grid3d(
        grid, template=template, where=where, property="raw"
    )
    assert ii[0, 0] == jj[0, 0] == -1
    assert np.isnan(top[0, 0]) and np.isnan(base[0, 0])
    np.testing.assert_allclose(top[~expected_mask], layer * 10)
    np.testing.assert_allclose(base[~expected_mask], (layer + 1) * 10)
    assert not inactive[~expected_mask].any()


@pytest.mark.parametrize("where", ["top", "base"])
@pytest.mark.parametrize("fformat", ["roff", "grdecl", "egrid"])
def test_nested_boundary_survives_export_import(
    tmp_path, nested_box_with_template, where, fformat
):
    """Read nested boundaries from ACTNUM after export and import."""
    _, merged, template = nested_box_with_template
    path = tmp_path / f"nested.{fformat}"
    merged.to_file(path, fformat=fformat)
    restored = xtgeo.grid_from_file(path, fformat=fformat)
    expected = xtgeo.surface_from_grid3d(merged, template=template, where=where)
    result = xtgeo.surface_from_grid3d(restored, template=template, where=where)
    np.testing.assert_array_equal(result.values.mask, expected.values.mask)
    np.testing.assert_allclose(result.values.compressed(), expected.values.compressed())
