"""Testing: test_grid_operations"""

import pathlib

import pytest
from hypothesis import given

import xtgeo

from .grid_generator import xtgeo_grids

EMEGFILE = pathlib.Path("3dgrids/eme/1/emerald_hetero_grid.roff")
EMERFILE = pathlib.Path("3dgrids/eme/1/emerald_hetero_region.roff")

EMEGFILE2 = pathlib.Path("3dgrids/eme/2/emerald_hetero_grid.roff")
EMEZFILE2 = pathlib.Path("3dgrids/eme/2/emerald_hetero.roff")

DUAL = pathlib.Path("3dgrids/etc/dual_distorted2.grdecl")
DUALPROPS = pathlib.Path("3dgrids/etc/DUAL")


def test_hybridgrid1(tmp_path, snapshot, helpers):
    grd = xtgeo.create_box_grid(
        (4, 3, 5),
        flip=1,
        oricenter=False,
        origin=(10.0, 20.0, 1000.0),
        rotation=30.0,
        increment=(100, 150, 5),
    )
    grd.subgrids = {
        "name1": [1],
        "name2": [2, 3],
        "name3": [4, 5],
    }
    assert grd.subgrids is not None  # initially, prior to subgrids

    grd.to_file(tmp_path / "test_hybridgrid1_asis.bgrdecl", fformat="bgrdecl")
    nhdiv = 40
    newnlay = grd.nlay * 2 + nhdiv
    snapshot.assert_match(
        helpers.df2csv(grd.get_dataframe(activeonly=False).tail(50).round()),
        "grid_pre_hybrid.csv",
    )
    grd.convert_to_hybrid(nhdiv=nhdiv, toplevel=1700, bottomlevel=1740)
    snapshot.assert_match(
        helpers.df2csv(grd.get_dataframe(activeonly=False).tail(50).round()),
        "grid_post_hybrid.csv",
    )

    assert grd.nlay == newnlay, "New NLAY number"
    assert grd.subgrids is None

    dzv = grd.get_dz()

    assert dzv.values.mean() == 5.0

    grd2 = xtgeo.grid_from_file(tmp_path / "test_hybridgrid1_asis.bgrdecl")
    snapshot.assert_match(
        helpers.df2csv(grd2.get_dataframe(activeonly=False).tail(50).round()),
        "grid_pre_hybrid.csv",
    )


def test_hybridgrid2(tmp_path, testdata_path):
    """Making a hybridgrid for Emerald case in region"""

    grd = xtgeo.create_box_grid((4, 3, 5))

    reg = xtgeo.gridproperty_from_file(testdata_path / EMERFILE, name="REGION")

    nhdiv = 40

    grd.convert_to_hybrid(
        nhdiv=nhdiv, toplevel=1650, bottomlevel=1690, region=reg, region_number=1
    )

    grd.to_file(tmp_path / "test_hybridgrid2.roff")


def test_inactivate_thin_cells(tmp_path, testdata_path):
    """Make hybridgrid for Emerald case in region, and inactive thin cells"""

    grd = xtgeo.grid_from_file(testdata_path / EMEGFILE)

    reg = xtgeo.gridproperty_from_file(testdata_path / EMERFILE, name="REGION")

    nhdiv = 40

    grd.convert_to_hybrid(
        nhdiv=nhdiv, toplevel=1650, bottomlevel=1690, region=reg, region_number=1
    )

    grd.inactivate_by_dz(0.001)

    grd.to_file(tmp_path / "test_hybridgrid2_inact_thin.roff")


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


def test_refine_vertically(testdata_path):
    """Do a grid refinement vertically."""

    emerald_grid = xtgeo.grid_from_file(testdata_path / EMEGFILE)
    assert emerald_grid.get_subgrids() == {"subgrid_0": 16, "subgrid_1": 30}

    avg_dz1 = emerald_grid.get_dz().values.mean()

    # idea; either a scalar (all cells), or a dictionary for zone wise
    emerald_grid.refine_vertically(3)

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


def test_reverse_row_axis_box(tmp_path):
    """Crop a grid."""

    grd = xtgeo.create_box_grid(
        origin=(1000, 4000, 300),
        increment=(100, 100, 2),
        dimension=(2, 3, 1),
        rotation=0,
    )

    assert grd.ijk_handedness == "left"
    grd.to_file(tmp_path / "reverse_left.grdecl", fformat="grdecl")
    grd.reverse_row_axis()
    assert grd.ijk_handedness == "right"
    grd.to_file(tmp_path / "reverse_right.grdecl", fformat="grdecl")


def test_reverse_row_axis_dual(tmp_path, testdata_path):
    """Reverse axis for distorted but small grid"""

    grd = xtgeo.grid_from_file(testdata_path / DUAL)

    assert grd.ijk_handedness == "left"
    grd.to_file(tmp_path / "dual_left.grdecl", fformat="grdecl")
    cellcorners1 = grd.get_xyz_cell_corners((5, 1, 1))
    grd.reverse_row_axis()
    assert grd.ijk_handedness == "right"
    grd.to_file(tmp_path / "dual_right.grdecl", fformat="grdecl")
    cellcorners2 = grd.get_xyz_cell_corners((5, 3, 1))

    assert cellcorners1[7] == cellcorners2[1]


def test_reverse_row_axis_dualprops(testdata_path):
    """Reverse axis for distorted but small grid with props"""

    grd = xtgeo.grid_from_file(
        testdata_path / DUALPROPS, fformat="eclipserun", initprops=["PORO", "PORV"]
    )

    poro = grd.gridprops.props[0]
    porowas = poro.copy()
    assert poro.values[1, 0, 0] == pytest.approx(0.17777, abs=0.01)
    assert grd.ijk_handedness == "left"

    grd.reverse_row_axis()
    assert poro.values[1, 2, 0] == pytest.approx(0.17777, abs=0.01)
    assert poro.values[1, 2, 0] == porowas.values[1, 0, 0]

    grd.reverse_row_axis()
    assert poro.values[1, 0, 0] == porowas.values[1, 0, 0]
    assert grd.ijk_handedness == "left"

    grd.reverse_row_axis(ijk_handedness="left")  # ie do nothing in this case
    assert poro.values[1, 0, 0] == porowas.values[1, 0, 0]
    assert grd.ijk_handedness == "left"


def test_reverse_row_axis_eme(tmp_path, testdata_path):
    """Reverse axis for emerald grid"""

    grd1 = xtgeo.grid_from_file(testdata_path / EMEGFILE)

    assert grd1.ijk_handedness == "left"
    grd1.to_file(tmp_path / "eme_left.roff", fformat="roff")
    grd2 = grd1.copy()
    geom1 = grd1.get_geometrics(return_dict=True)
    grd2.reverse_row_axis()
    assert grd2.ijk_handedness == "right"
    grd2.to_file(tmp_path / "eme_right.roff", fformat="roff")
    geom2 = grd2.get_geometrics(return_dict=True)

    assert geom1["avg_rotation"] == pytest.approx(geom2["avg_rotation"], abs=0.01)


def test_copy_grid(tmp_path, testdata_path):
    """Copy a grid."""

    grd = xtgeo.grid_from_file(testdata_path / EMEGFILE2)
    grd2 = grd.copy()

    grd.to_file(tmp_path / "gcp1.roff")
    grd2.to_file(tmp_path / "gcp2.roff")

    xx1 = xtgeo.grid_from_file(tmp_path / "gcp1.roff")
    xx2 = xtgeo.grid_from_file(tmp_path / "gcp2.roff")

    assert xx1._zcornsv.mean() == xx2._zcornsv.mean()
    assert xx1._actnumsv.mean() == xx2._actnumsv.mean()


def test_crop_grid(tmp_path, testdata_path):
    """Crop a grid."""

    grd = xtgeo.grid_from_file(testdata_path / EMEGFILE2)
    zprop = xtgeo.gridproperty_from_file(
        testdata_path / EMEZFILE2, name="Zone", grid=grd
    )

    grd.crop((30, 60), (20, 40), (1, 46), props=[zprop])

    grd.to_file(tmp_path / "grid_cropped.roff")

    grd2 = xtgeo.grid_from_file(tmp_path / "grid_cropped.roff")

    assert grd2.ncol == 31


def test_crop_grid_after_copy(testdata_path):
    """Copy a grid, then crop and check number of active cells."""

    grd = xtgeo.grid_from_file(testdata_path / EMEGFILE2)
    grd.describe()
    grd.describe(details=True)

    grd2 = grd.copy()
    grd2.describe(details=True)

    assert grd.propnames == grd2.propnames

    grd2.crop((1, 30), (40, 80), (23, 46))

    grd2.describe(details=True)


@given(xtgeo_grids)
def test_reduce_to_one_layer(grd):
    grd.reduce_to_one_layer()

    assert grd.nlay == 1


def test_get_boundary_polygons_grid_outline(testdata_path):
    """Test getting a boundary for a grid."""

    grid = xtgeo.grid_from_file(testdata_path / EMEGFILE)

    boundary = grid.get_boundary_polygons(simplify=False)
    df = boundary.get_dataframe(copy=False)

    assert df["POLY_ID"].nunique() == 1
    assert df[boundary.yname].min() == pytest.approx(5930003, abs=2)
    assert df[boundary.yname].max() == pytest.approx(5937505, abs=2)
    assert df[boundary.xname].min() == pytest.approx(459078, abs=2)
    assert df[boundary.xname].max() == pytest.approx(466984, abs=2)


def test_get_boundary_polygons_grid_region(testdata_path):
    """Test getting a boundary for a region in a grid."""

    grid = xtgeo.grid_from_file(testdata_path / EMEGFILE)
    reg = xtgeo.gridproperty_from_file(testdata_path / EMERFILE, name="REGION")

    boundary = grid.get_boundary_polygons(
        simplify=False, filter_array=(reg.values == 1)
    )
    df = boundary.get_dataframe(copy=False)

    assert df["POLY_ID"].nunique() == 1
    assert df[boundary.yname].min() == pytest.approx(5932092, abs=2)
    assert df[boundary.yname].max() == pytest.approx(5936223, abs=2)
    assert df[boundary.xname].min() == pytest.approx(460344, abs=2)
    assert df[boundary.xname].max() == pytest.approx(463765, abs=2)
