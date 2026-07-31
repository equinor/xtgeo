"""Testing: test_grid_refinement"""

import pathlib

import numpy as np
import pytest

import xtgeo

EMEGFILE = pathlib.Path("3dgrids/eme/1/emerald_hetero_grid.roff")
EMERFILE = pathlib.Path("3dgrids/eme/1/emerald_hetero_region.roff")

EMEGFILE2 = pathlib.Path("3dgrids/eme/2/emerald_hetero_grid.roff")
EMEZFILE2 = pathlib.Path("3dgrids/eme/2/emerald_hetero.roff")


def test_refine(testdata_path):
    """Do a grid refinement in all direction."""

    grid = xtgeo.create_box_grid(
        (100, 100, 50), increment=(100, 100, 20), rotation=45.0
    )

    avg_dx1 = grid.get_dx().values.mean()
    avg_dy1 = grid.get_dy().values.mean()
    avg_dz1 = grid.get_dz().values.mean()

    refine_x = 2
    refine_y = 2
    refine_z = 3

    # idea; either a scalar (all cells), or a dictionary for zone wise
    grid.refine(refine_x, refine_y, refine_z)

    avg_dx2 = grid.get_dx().values.mean()
    avg_dy2 = grid.get_dy().values.mean()
    avg_dz2 = grid.get_dz().values.mean()

    assert avg_dx1 == pytest.approx(refine_x * avg_dx2, abs=0.0001)
    assert avg_dy1 == pytest.approx(refine_y * avg_dy2, abs=0.0001)
    assert avg_dz1 == pytest.approx(refine_z * avg_dz2, abs=0.0001)


def test_refine_with_attached_props(testdata_path):
    """Do a grid refinement in all direction, grid with attached props."""

    grid = xtgeo.create_box_grid((30, 30, 10), increment=(100, 100, 20), rotation=45.0)

    discrete = xtgeo.GridProperty(grid, name="DISCRETE", discrete=True, values=1)
    continuous = xtgeo.GridProperty(grid, name="CONTINUOUS", discrete=False, values=0.5)

    assert discrete.dimensions == (30, 30, 10)
    assert continuous.dimensions == (30, 30, 10)
    assert grid.propnames == ["DISCRETE", "CONTINUOUS"]

    refine_x = 2
    refine_y = 2
    refine_z = 3

    grid.refine(refine_x, refine_y, refine_z)
    assert grid.dimensions == (60, 60, 30)
    assert grid.propnames == ["DISCRETE", "CONTINUOUS"]
    dprop = grid.get_prop_by_name("DISCRETE")
    assert dprop.dimensions == (60, 60, 30)
    cprop = grid.get_prop_by_name("CONTINUOUS")
    assert cprop.dimensions == (60, 60, 30)
    # Check that geometry is properly linked
    assert dprop.geometry is grid
    assert cprop.geometry is grid


def test_refine_vertically_with_attached_props(testdata_path):
    """Do a vertical grid refinement, grid with attached props."""

    grid = xtgeo.create_box_grid((30, 30, 10), increment=(100, 100, 20), rotation=45.0)

    discrete = xtgeo.GridProperty(grid, name="DISCRETE", discrete=True, values=1)
    continuous = xtgeo.GridProperty(grid, name="CONTINUOUS", discrete=False, values=0.5)

    assert discrete.dimensions == (30, 30, 10)
    assert continuous.dimensions == (30, 30, 10)
    assert grid.propnames == ["DISCRETE", "CONTINUOUS"]

    refine_z = 3

    grid.refine_vertically(refine_z)
    assert grid.dimensions == (30, 30, 30)
    assert grid.propnames == ["DISCRETE", "CONTINUOUS"]
    dprop = grid.get_prop_by_name("DISCRETE")
    assert dprop.dimensions == (30, 30, 30)
    cprop = grid.get_prop_by_name("CONTINUOUS")
    assert cprop.dimensions == (30, 30, 30)
    # Check that geometry is properly linked
    assert dprop.geometry is grid
    assert cprop.geometry is grid


def test_refine_with_factor_one_and_attached_props(testdata_path):
    """Test refinement with factor 1 (no actual refinement) with attached props."""

    grid = xtgeo.create_box_grid((10, 10, 5), increment=(100, 100, 20))

    _ = xtgeo.GridProperty(grid, name="DISCRETE", discrete=True, values=1)
    _ = xtgeo.GridProperty(grid, name="CONTINUOUS", discrete=False, values=0.5)

    # Refine with factor 1 (should result in no change)
    grid.refine(1, 1, 1)

    assert grid.dimensions == (10, 10, 5)
    assert grid.propnames == ["DISCRETE", "CONTINUOUS"]
    dprop = grid.get_prop_by_name("DISCRETE")
    assert dprop.dimensions == (10, 10, 5)
    assert dprop.geometry is grid
    cprop = grid.get_prop_by_name("CONTINUOUS")
    assert cprop.geometry is grid


def test_refine_single_direction_with_attached_props(testdata_path):
    """Test refinement in only one direction with attached props."""

    grid = xtgeo.create_box_grid((10, 10, 5), increment=(100, 100, 20))

    _ = xtgeo.GridProperty(grid, name="DISCRETE", discrete=True, values=1)

    # Refine only in Z direction
    grid.refine(1, 1, 2)

    assert grid.dimensions == (10, 10, 10)
    assert grid.propnames == ["DISCRETE"]
    dprop = grid.get_prop_by_name("DISCRETE")
    assert dprop.dimensions == (10, 10, 10)
    assert dprop.geometry is grid


def test_refine_lateral_with_dict(testdata_path):
    """Do lateral grid refinement from i = 41 - 60, j = 41 - 60 with factor 2"""

    grid = xtgeo.create_box_grid(
        (100, 100, 50), increment=(100, 100, 20), rotation=45.0
    )

    avg_dx1 = grid.get_dx().values[40:60, 40:60, :].mean()
    avg_dy1 = grid.get_dy().values[40:60, 40:60, :].mean()

    refinement = 2

    refine_factor = dict.fromkeys(range(41, 61), refinement)

    grid.refine(refine_factor, refine_factor, 1)

    avg_dx2 = grid.get_dx().values[40:80, 40:80, :].mean()
    avg_dy2 = grid.get_dy().values[40:80, 40:80, :].mean()

    assert avg_dx1 == pytest.approx(refinement * avg_dx2, abs=0.0001)
    assert avg_dy1 == pytest.approx(refinement * avg_dy2, abs=0.0001)


def test_refine_lateral_with_dict_and_attached_props(testdata_path):
    """Do lateral grid refinement with dict and attached props."""

    grid = xtgeo.create_box_grid((30, 20, 5), increment=(100, 100, 20))

    prop = xtgeo.GridProperty(grid, name="PORO", discrete=False, values=0.25)

    # Refine cells 11-20 with factor 2, others get default factor 1
    refine_factor = dict.fromkeys(range(11, 21), 2)

    grid.refine(refine_factor, 1, 1)

    # Should have 10 cells unchanged + 10 cells refined to 20 + 10 cells
    # unchanged = 40 columns
    assert grid.dimensions == (40, 20, 5)
    assert grid.propnames == ["PORO"]
    prop = grid.get_prop_by_name("PORO")
    assert prop.dimensions == (40, 20, 5)
    assert prop.geometry is grid


def test_refine_vertically(testdata_path):
    """Do a grid refinement vertically."""

    emerald_grid = xtgeo.grid_from_file(testdata_path / EMEGFILE)
    assert emerald_grid.get_subgrids() == {"subgrid_0": 16, "subgrid_1": 30}

    avg_dz1 = emerald_grid.get_dz().values.mean()

    emerald_grid.append_prop(
        xtgeo.gridproperty_from_file(testdata_path / EMERFILE, name="REGION")
    )

    df1 = emerald_grid.get_dataframe()

    # idea; either a scalar (all cells), or a dictionary for zone wise
    emerald_grid.refine_vertically(3)

    df2 = emerald_grid.get_dataframe()

    assert df1["REGION"].mean() == pytest.approx(df2["REGION"].mean(), rel=1e-6)

    avg_dz2 = emerald_grid.get_dz().values.mean()

    assert avg_dz1 == pytest.approx(3 * avg_dz2, abs=0.0001)

    assert emerald_grid.get_subgrids() == {"subgrid_0": 48, "subgrid_1": 90}
    emerald_grid.inactivate_by_dz(0.001)


def test_refine_vertically_per_zone(testdata_path):
    """Do a grid refinement vertically, via a dict per zone."""

    emerald2_grid = xtgeo.grid_from_file(testdata_path / EMEGFILE2)
    grd = emerald2_grid.copy()
    emerald2_zone = xtgeo.gridproperty_from_file(
        testdata_path / EMEZFILE2, grid=grd, name="Zone"
    )

    assert emerald2_zone.values.min() == 1
    assert emerald2_zone.values.max() == 2

    assert grd.subgrids == {"subgrid_0": range(1, 17), "subgrid_1": range(17, 47)}

    refinement = {1: 4, 2: 2}
    grd.refine_vertically(refinement, zoneprop=emerald2_zone)

    assert grd.get_subgrids() == {"Zone1": 64, "Zone2": 60}

    grd = emerald2_grid.copy()
    grd.refine_vertically(refinement)  # no zoneprop

    assert grd.get_subgrids() == {"subgrid_0": 64, "subgrid_1": 60}


def test_refine_lateral_with_list_even_split():
    """Refine I/J using a list of per-section factors with an even split."""

    grid = xtgeo.create_box_grid((9, 6, 4), increment=(10, 10, 5))
    avg_dx1 = grid.get_dx().values.mean()
    avg_dy1 = grid.get_dy().values.mean()

    # 9 cols / 3 sections -> 3 cols per section, factors [1, 2, 3]
    # 6 rows / 2 sections -> 3 rows per section, factors [2, 4]
    grid.refine([1, 2, 3], [2, 4], 1)

    expected_ncol = 3 * 1 + 3 * 2 + 3 * 3
    expected_nrow = 3 * 2 + 3 * 4
    assert grid.dimensions == (expected_ncol, expected_nrow, 4)

    # The total physical extent in X/Y must be preserved, so the mean cell
    # size scales like ncol_before / ncol_after.
    avg_dx2 = grid.get_dx().values.mean()
    avg_dy2 = grid.get_dy().values.mean()
    assert avg_dx2 == pytest.approx(avg_dx1 * 9 / expected_ncol, abs=1e-6)
    assert avg_dy2 == pytest.approx(avg_dy1 * 6 / expected_nrow, abs=1e-6)


def test_refine_lateral_with_list_uneven_split():
    """List sections that don't divide evenly should follow numpy.array_split."""

    grid = xtgeo.create_box_grid((10, 7, 2), increment=(10, 10, 5))

    # 10 cols / 3 sections -> sizes 4, 3, 3 (extras to earlier sections)
    # 7 rows / 4 sections -> sizes 2, 2, 2, 1
    grid.refine([2, 1, 3], [1, 2, 1, 2], 1)

    expected_ncol = 4 * 2 + 3 * 1 + 3 * 3
    expected_nrow = 2 * 1 + 2 * 2 + 2 * 1 + 1 * 2
    assert grid.dimensions == (expected_ncol, expected_nrow, 2)


def test_refine_lateral_with_list_and_attached_props():
    """List-based refinement must keep attached properties in sync."""

    grid = xtgeo.create_box_grid((6, 4, 3), increment=(10, 10, 5))
    discrete = xtgeo.GridProperty(grid, name="ZONE", discrete=True, values=1)
    # Tag each I-section with a distinct value so we can verify ordering after
    # refinement.
    discrete.values[0:2, :, :] = 10
    discrete.values[2:4, :, :] = 20
    discrete.values[4:6, :, :] = 30

    # 6 cols / 3 sections -> 2 cols per section, factors [1, 2, 3]
    # 4 rows kept identical via scalar 1
    grid.refine([1, 2, 3], 1, 1)

    expected_ncol = 2 * 1 + 2 * 2 + 2 * 3
    assert grid.dimensions == (expected_ncol, 4, 3)

    zone = grid.get_prop_by_name("ZONE")
    assert zone.dimensions == (expected_ncol, 4, 3)
    assert zone.geometry is grid

    vals_i = zone.values[:, 0, 0].tolist()
    # First section (factor 1) keeps 2 cells of value 10,
    # second section (factor 2) yields 2*2 = 4 cells of value 20,
    # third section (factor 3) yields 2*3 = 6 cells of value 30.
    assert vals_i == [10] * 2 + [20] * 4 + [30] * 6


def test_refine_lateral_with_tuple_is_equivalent_to_list():
    """A tuple of factors must behave the same as the equivalent list."""

    grid_list = xtgeo.create_box_grid((8, 4, 2), increment=(10, 10, 5))
    grid_tuple = grid_list.copy()

    grid_list.refine([1, 2, 1, 2], [1, 3], 1)
    grid_tuple.refine((1, 2, 1, 2), (1, 3), 1)

    assert grid_list.dimensions == grid_tuple.dimensions
    np.testing.assert_allclose(grid_list.get_dx().values, grid_tuple.get_dx().values)
    np.testing.assert_allclose(grid_list.get_dy().values, grid_tuple.get_dy().values)


def test_refine_lateral_list_single_section_equivalent_to_scalar():
    """A single-element list should match the equivalent scalar refinement."""

    base = xtgeo.create_box_grid((5, 5, 2), increment=(10, 10, 5))
    g_scalar = base.copy()
    g_list = base.copy()

    g_scalar.refine(3, 2, 1)
    g_list.refine([3], [2], 1)

    assert g_scalar.dimensions == g_list.dimensions
    np.testing.assert_allclose(g_scalar.get_dx().values, g_list.get_dx().values)
    np.testing.assert_allclose(g_scalar.get_dy().values, g_list.get_dy().values)


def test_refine_lateral_list_only_one_direction():
    """Passing a list in one direction and scalar in the other must work."""

    grid = xtgeo.create_box_grid((6, 5, 2), increment=(10, 10, 5))
    avg_dy1 = grid.get_dy().values.mean()

    grid.refine([1, 2, 3], 1, 1)

    expected_ncol = 2 * 1 + 2 * 2 + 2 * 3
    assert grid.dimensions == (expected_ncol, 5, 2)
    # J direction (and its dy) must be untouched
    assert grid.get_dy().values.mean() == pytest.approx(avg_dy1, abs=1e-6)


def test_refine_lateral_range_is_supported():
    """Any non-text sequence should work for the lateral section form."""

    grid_range = xtgeo.create_box_grid((6, 4, 2), increment=(10, 10, 5))
    grid_list = grid_range.copy()

    grid_range.refine(range(1, 4), range(1, 3), 1)
    grid_list.refine([1, 2, 3], [1, 2], 1)

    assert grid_range.dimensions == grid_list.dimensions
    np.testing.assert_allclose(grid_range.get_dx().values, grid_list.get_dx().values)
    np.testing.assert_allclose(grid_range.get_dy().values, grid_list.get_dy().values)


@pytest.mark.parametrize(
    "refine_col, refine_row",
    [
        ({1: 2, "a": 3}, 1),
        (1, {1: 2, "b": 3}),
        ({1: True}, 1),
        (1, {1: True}),
    ],
)
def test_refine_lateral_dict_validation_errors(refine_col, refine_row):
    """Dict keys and values must be validated with clear errors."""

    grid = xtgeo.create_box_grid((5, 5, 2), increment=(10, 10, 5))
    with pytest.raises(TypeError):
        grid.refine(refine_col, refine_row, 1)


@pytest.mark.parametrize(
    "refine_col, refine_row",
    [
        ({0: 2}, 1),
        ({6: 2}, 1),  # ncol=5
        (1, {0: 2}),
        (1, {6: 2}),  # nrow=5
    ],
)
def test_refine_lateral_dict_out_of_range_indices(refine_col, refine_row):
    """Out-of-range lateral dict indices must be rejected."""

    grid = xtgeo.create_box_grid((5, 5, 2), increment=(10, 10, 5))
    with pytest.raises(ValueError):
        grid.refine(refine_col, refine_row, 1)


def test_refine_vertically_rejects_bool():
    """Vertical refinement must reject bool inputs explicitly."""

    grid = xtgeo.create_box_grid((5, 5, 4), increment=(10, 10, 5))

    with pytest.raises(TypeError):
        grid.refine_vertically(True)

    with pytest.raises(TypeError):
        grid.refine_vertically({1: True})


def test_refine_vertical_scalar_zero_rejected():
    """Zero refinement must be rejected for vertical scalar APIs."""

    grid = xtgeo.create_box_grid((5, 5, 4), increment=(10, 10, 5))

    with pytest.raises(ValueError):
        grid.refine(1, 1, 0)

    with pytest.raises(ValueError):
        grid.refine_vertically(0)


@pytest.mark.parametrize(
    "refine_col, refine_row, expected_exc",
    [
        (0, 1, ValueError),
        (1, 0, ValueError),
        ([], 1, ValueError),
        (1, (), ValueError),
        ([1, 2, 3, 4, 5, 6, 7], 1, ValueError),  # more sections than ncol=5
        (1, [1, 2, 3, 4, 5, 6, 7], ValueError),  # more sections than nrow=5
        ([0, 2], 1, ValueError),
        ([1, -1], 1, ValueError),
        ([1.5, 2], 1, (TypeError, ValueError)),
        (1, ["a", "b"], (TypeError, ValueError)),
    ],
)
def test_refine_lateral_list_validation_errors(refine_col, refine_row, expected_exc):
    """Invalid list inputs must raise a clear error."""

    grid = xtgeo.create_box_grid((5, 5, 2), increment=(10, 10, 5))
    with pytest.raises(expected_exc):
        grid.refine(refine_col, refine_row, 1)


def test_refine_lateral_list_rejects_for_layer():
    """A list passed to refine_layer is not supported and must error."""

    grid = xtgeo.create_box_grid((5, 5, 4), increment=(10, 10, 5))
    with pytest.raises(TypeError):
        grid.refine(1, 1, [1, 2])
