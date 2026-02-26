"""Testing: test_grid_refinement"""

import pathlib

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
