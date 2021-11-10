"""
Test xtgeo against emerald grid residing in xtgeo-testdata
"""
from collections import OrderedDict
from os.path import join

import numpy as np
import pytest

import xtgeo


@pytest.fixture()
def emerald_grid(testpath):
    return xtgeo.grid_from_file(
        join(testpath, "3dgrids/eme/1/emerald_hetero_grid.roff")
    )


@pytest.fixture()
def emerald_region(testpath):
    return xtgeo.gridproperty_from_file(
        join(testpath, "3dgrids/eme/1/emerald_hetero_region.roff"),
        name="REGION",
    )


@pytest.fixture()
def emerald2_grid(testpath):
    return xtgeo.grid_from_file(
        join(testpath, "3dgrids/eme/2/emerald_hetero_grid.roff")
    )


@pytest.fixture()
def emerald_property_file(testpath):
    return join(testpath, "3dgrids/eme/1/emerald_hetero.roff")


@pytest.fixture()
def emerald2_property_file(testpath):
    return join(testpath, "3dgrids/eme/2/emerald_hetero.roff")


@pytest.fixture()
def emerald2_zone(emerald2_property_file):
    return xtgeo.gridproperty_from_file(emerald2_property_file, name="Zone")


def test_emerald_grid_properties(emerald_grid):
    assert isinstance(emerald_grid, xtgeo.Grid)
    assert emerald_grid.name == "emerald_hetero_grid"
    assert emerald_grid.dimensions == (70, 100, 46)
    assert emerald_grid.subgrids == OrderedDict(
        [("subgrid_0", range(1, 17)), ("subgrid_1", range(17, 47))]
    )


def test_roffbin_get_dataframe_for_grid(emerald_grid):
    """Import ROFF grid and return a grid dataframe (no props)."""
    df = emerald_grid.dataframe()
    print(df.head())

    assert len(df) == emerald_grid.nactive

    assert df["X_UTME"][0] == pytest.approx(459176.7937727844, abs=0.1)

    assert len(df.columns) == 6

    df = emerald_grid.dataframe(activeonly=False)

    assert len(df.columns) == 7
    assert len(df) != emerald_grid.nactive

    assert len(df) == np.prod(emerald_grid.dimensions)


def test_subgrids(emerald_grid):
    """Import ROFF and test different subgrid functions."""

    newsub = OrderedDict()
    newsub["XX1"] = 20
    newsub["XX2"] = 2
    newsub["XX3"] = 24

    emerald_grid.set_subgrids(newsub)

    subs = emerald_grid.get_subgrids()

    assert subs == newsub

    _i_index, _j_index, k_index = emerald_grid.get_ijk()

    zprop = k_index.copy()
    zprop.values[k_index.values > 4] = 2
    zprop.values[k_index.values <= 4] = 1
    emerald_grid.subgrids_from_zoneprop(zprop)

    # rename
    emerald_grid.rename_subgrids(["AAAA", "BBBB"])
    assert "AAAA" in emerald_grid.subgrids.keys()

    # set to None
    emerald_grid.subgrids = None
    assert emerald_grid._subgrids is None


def test_activate_all_cells(tmp_path, emerald_grid):
    """Make the grid active for all cells."""
    emerald_grid.activate_all()
    assert emerald_grid.nactive == emerald_grid.ntotal
    emerald_grid.to_file(tmp_path / "emerald_all_active.roff")


def test_get_adjacent_cells(tmp_path, emerald_grid):
    """Get the cell indices for discrete value X vs Y, if connected."""
    actnum = emerald_grid.get_actnum()
    actnum.to_file(tmp_path / "emerald_actnum.roff")
    result = emerald_grid.get_adjacent_cells(actnum, 0, 1, activeonly=False)
    result.to_file(tmp_path / "emerald_adj_cells.roff")


def test_roffbin_import1(emerald_grid):
    """Test roff binary import case 1."""

    # get dZ...
    dzv = emerald_grid.get_dz()

    dzval = dzv.values
    # get the value is cell 32 73 1 shall be 2.761
    mydz = float(dzval[31:32, 72:73, 0:1])
    assert mydz == pytest.approx(2.761, abs=0.001), "Grid DZ Emerald"

    # get dX dY
    dxv, dyv = emerald_grid.get_dxdy()

    mydx = float(dxv.values3d[31:32, 72:73, 0:1])
    mydy = float(dyv.values3d[31:32, 72:73, 0:1])

    assert mydx == pytest.approx(118.51, abs=0.01), "Grid DX Emerald"
    assert mydy == pytest.approx(141.26, abs=0.01), "Grid DY Emerald"

    # get X Y Z coordinates (as GridProperty objects) in one go
    xvv, yvv, zvv = emerald_grid.get_xyz(names=["xxx", "yyy", "zzz"])

    assert xvv.name == "xxx", "Name of X coord"
    xvv.name = "Xerxes"

    # attach some properties to grid
    emerald_grid.props = [xvv, yvv]

    assert emerald_grid.get_prop_by_name("Xerxes").name == "Xerxes"


def test_grid_design(emerald_grid):
    """Determine if a subgrid is topconform (T), baseconform (B), proportional (P).

    "design" refers to type of conformity
    "dzsimbox" is avg or representative simbox thickness per cell

    """

    assert emerald_grid.subgrids == OrderedDict(
        [("subgrid_0", range(1, 17)), ("subgrid_1", range(17, 47))]
    )

    code = emerald_grid.estimate_design(1)
    assert code["design"] == "P"
    assert code["dzsimbox"] == pytest.approx(2.5488, abs=0.001)

    code = emerald_grid.estimate_design(2)
    assert code["design"] == "T"
    assert code["dzsimbox"] == pytest.approx(3.0000, abs=0.001)

    code = emerald_grid.estimate_design("subgrid_0")
    assert code["design"] == "P"

    code = emerald_grid.estimate_design("subgrid_1")
    assert code["design"] == "T"

    code = emerald_grid.estimate_design("subgrid_2")
    assert code is None

    with pytest.raises(ValueError):
        code = emerald_grid.estimate_design(nsub=None)


def test_flip(emerald_grid):
    """Determine if grid is flipped (lefthanded vs righthanded)."""

    assert emerald_grid.estimate_flip() == 1

    emerald_grid.create_box(dimension=(30, 20, 3), flip=-1)
    assert emerald_grid.estimate_flip() == -1

    emerald_grid.create_box(dimension=(30, 20, 3), rotation=30, flip=-1)
    assert emerald_grid.estimate_flip() == -1

    emerald_grid.create_box(dimension=(30, 20, 3), rotation=190, flip=-1)
    assert emerald_grid.estimate_flip() == -1


def test_gridquality_properties_emeg(show_plot, emerald_grid):
    quality_props = emerald_grid.get_gridquality_properties()

    concp = quality_props.get_prop_by_name("concave_proj")
    if show_plot:
        lay = 23
        layslice = xtgeo.plot.Grid3DSlice()
        layslice.canvas(title=f"Layer {lay}")
        layslice.plot_gridslice(
            emerald_grid,
            prop=concp,
            mode="layer",
            index=lay + 1,
            window=None,
            linecolor="black",
        )

        layslice.show()


def test_roffbin_import2(emerald_property_file):
    """Import roffbin, with several props in one file."""

    dz = xtgeo.gridproperty_from_file(
        emerald_property_file, fformat="roff", name="Z_increment"
    )

    assert dz.values.dtype == np.float64
    assert dz.values.mean() == pytest.approx(2.635, abs=0.001)

    hc = xtgeo.gridproperty_from_file(
        emerald_property_file, fformat="roff", name="Oil_HCPV"
    )

    assert hc.values.dtype == np.float64
    assert hc.values3d.shape == (70, 100, 46)
    _ncol, nrow, _nlay = hc.values3d.shape

    assert nrow == 100, "NROW from shape (Emerald)"

    assert hc.values.mean() == pytest.approx(1446.4611912446985, abs=0.0001)


def test_hybridgrid2(tmpdir, emerald_grid, emerald_region):
    """Making a hybridgrid for Emerald case in region"""

    nhdiv = 40

    emerald_grid.convert_to_hybrid(
        nhdiv=nhdiv,
        toplevel=1650,
        bottomlevel=1690,
        region=emerald_region,
        region_number=1,
    )

    emerald_grid.to_file(join(tmpdir, "test_hybridgrid2.roff"))


def test_inactivate_thin_cells(tmpdir, emerald_grid, emerald_region):
    """Make hybridgrid for Emerald case in region, and inactive thin cells"""

    nhdiv = 40

    emerald_grid.convert_to_hybrid(
        nhdiv=nhdiv,
        toplevel=1650,
        bottomlevel=1690,
        region=emerald_region,
        region_number=1,
    )

    emerald_grid.inactivate_by_dz(0.001)

    emerald_grid.to_file(join(tmpdir, "test_hybridgrid2_inact_thin.roff"))


def test_refine_vertically(tmpdir, emerald_grid):
    """Do a grid refinement vertically."""

    assert emerald_grid.get_subgrids() == OrderedDict(
        [("subgrid_0", 16), ("subgrid_1", 30)]
    )

    avg_dz1 = emerald_grid.get_dz().values3d.mean()

    # idea; either a scalar (all cells), or a dictionary for zone wise
    emerald_grid.refine_vertically(3)

    avg_dz2 = emerald_grid.get_dz().values3d.mean()

    assert avg_dz1 == pytest.approx(3 * avg_dz2, abs=0.0001)

    assert emerald_grid.get_subgrids() == OrderedDict(
        [("subgrid_0", 48), ("subgrid_1", 90)]
    )
    emerald_grid.inactivate_by_dz(0.001)


def test_refine_vertically_per_zone(tmpdir, emerald2_grid, emerald2_zone):
    """Do a grid refinement vertically, via a dict per zone."""

    grd = emerald2_grid.copy()

    assert emerald2_zone.values.min() == 1
    assert emerald2_zone.values.max() == 2

    assert grd.subgrids == OrderedDict(
        [("subgrid_0", range(1, 17)), ("subgrid_1", range(17, 47))]
    )

    refinement = {1: 4, 2: 2}
    grd.refine_vertically(refinement, zoneprop=emerald2_zone)

    assert grd.get_subgrids() == OrderedDict([("zone1", 64), ("zone2", 60)])

    grd = emerald2_grid.copy()
    grd.refine_vertically(refinement)  # no zoneprop

    assert grd.get_subgrids() == OrderedDict([("subgrid_0", 64), ("subgrid_1", 60)])


def test_reverse_row_axis_eme(tmpdir, emerald_grid):
    """Reverse axis for emerald grid"""

    assert emerald_grid.ijk_handedness == "left"
    emerald_grid.to_file(join(tmpdir, "eme_left.roff"), fformat="roff")
    geom1 = emerald_grid.get_geometrics(return_dict=True)

    emerald_grid2 = emerald_grid.copy()

    emerald_grid2.reverse_row_axis()
    assert emerald_grid2.ijk_handedness == "right"
    emerald_grid2.to_file(join(tmpdir, "eme_right.roff"), fformat="roff")
    geom2 = emerald_grid2.get_geometrics(return_dict=True)

    assert geom1["avg_rotation"] == pytest.approx(geom2["avg_rotation"], abs=0.01)


def test_crop_grid(tmpdir, emerald2_grid, emerald2_zone):

    emerald2_grid.crop((30, 60), (20, 40), (1, 46), props=[emerald2_zone])

    assert emerald2_grid.ncol == 31
    assert emerald2_grid.nrow == 21
    assert emerald2_grid.nlay == 46

    assert emerald2_grid.get_actnum().values.shape == (31, 21, 46)
