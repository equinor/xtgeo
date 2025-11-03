"""Tests for 3D grid."""

import logging
import math
import pathlib
import warnings

import numpy as np
import pytest
from hypothesis import given

import xtgeo
from xtgeo.common.log import functimer
from xtgeo.grid3d import Grid

from .grid_generator import dimensions, increments, xtgeo_grids

logger = logging.getLogger(__name__)

REEKFILE = pathlib.Path("3dgrids/reek/REEK.EGRID")
REEKFIL2 = pathlib.Path("3dgrids/reek3/reek_sim.grdecl")  # ASCII GRDECL
REEKFIL3 = pathlib.Path("3dgrids/reek3/reek_sim.bgrdecl")  # binary GRDECL
REEKFIL4 = pathlib.Path("3dgrids/reek/reek_geo_grid.roff")
REEKFIL5 = pathlib.Path("3dgrids/reek/reek_geo2_grid_3props.roff")
# brilfile = '../xtgeo-testdata/3dgrids/bri/B.GRID' ...disabled
BRILGRDECL = pathlib.Path("3dgrids/bri/b.grdecl")
BANAL6 = pathlib.Path("3dgrids/etc/banal6.roff")
B7 = pathlib.Path("3dgrids/etc/banal7_grid_params.roff")
B9 = pathlib.Path("3dgrids/etc/b9.roff")
GRIDQC1 = pathlib.Path("3dgrids/etc/gridqc1.roff")
GRIDQC1_CELLVOL = pathlib.Path("3dgrids/etc/gridqc1_totbulk.roff")
GRIDQC2 = pathlib.Path("3dgrids/etc/gridqc_negthick_twisted.roff")

DUALFIL1 = pathlib.Path("3dgrids/etc/dual_grid.roff")
DUALFIL3 = pathlib.Path("3dgrids/etc/TEST_DPDK.EGRID")

EME1 = pathlib.Path("3dgrids/eme/2/eme_small_w_hole_grid_params.roff")
EME1PROP = pathlib.Path("3dgrids/eme/2/eme_small_w_hole_grid_params.roff")


@pytest.fixture()
def emerald_grid(testdata_path):
    return xtgeo.grid_from_file(
        testdata_path / pathlib.Path("3dgrids/eme/1/emerald_hetero_grid.roff")
    )


def test_import_wrong_format(tmp_path):
    grd = xtgeo.create_box_grid((2, 2, 2))
    grd.to_file(tmp_path / "grd.roff")
    with pytest.raises(ValueError):
        xtgeo.grid_from_file(tmp_path / "grd.roff", fformat="stupid_wrong_name")


def test_get_set_name():
    grd = xtgeo.create_box_grid((2, 2, 2))
    grd.name = "xxx"
    assert grd.name == "xxx"


def test_create_shoebox(tmp_path):
    """Make a shoebox grid from scratch."""
    grd = xtgeo.create_box_grid((4, 3, 5))
    grd.to_file(tmp_path / "shoebox_default.roff")

    grd = xtgeo.create_box_grid((2, 3, 4), flip=-1)
    grd.to_file(tmp_path / "shoebox_default_flipped.roff")

    @functimer(output="info", comment="Large box grid created, with 1.8 mill cells")
    def box_grid_large_create():
        _ = xtgeo.create_box_grid((300, 200, 30), increment=(20, 20, 1))

    box_grid_large_create()

    grd = xtgeo.create_box_grid(
        origin=(0, 0, 1000), dimension=(300, 200, 30), increment=(20, 20, 1), flip=-1
    )

    dx, dy = (grd.get_dx(), grd.get_dy())

    assert dx.values.mean() == pytest.approx(20.0, abs=0.0001)
    assert dy.values.mean() == pytest.approx(20.0, abs=0.0001)

    grd = xtgeo.create_box_grid(
        origin=(0, 0, 1000), dimension=(30, 30, 3), rotation=45, increment=(20, 20, 1)
    )

    x, y, z = grd.get_xyz()

    assert x.values1d[0] == pytest.approx(0.0, abs=0.001)
    assert y.values1d[0] == pytest.approx(20 * math.cos(45 * math.pi / 180), abs=0.001)
    assert z.values1d[0] == pytest.approx(1000.5, abs=0.001)

    grd = xtgeo.create_box_grid(
        origin=(0, 0, 1000),
        dimension=(30, 30, 3),
        rotation=45,
        increment=(20, 20, 1),
        oricenter=True,
    )

    x, y, z = grd.get_xyz()

    assert x.values1d[0] == pytest.approx(0.0, abs=0.001)
    assert y.values1d[0] == pytest.approx(0.0, abs=0.001)
    assert z.values1d[0] == pytest.approx(1000.0, abs=0.001)


@pytest.mark.parametrize(
    "dimensions",
    [
        (100, 1, 1),
        (100, 1, 20),
        (300, 20, 30),
    ],
)
def test_shoebox_egrid(tmp_path, dimensions):
    grd = xtgeo.create_box_grid(dimension=dimensions)
    grd.to_file(tmp_path / "E1.EGRID", fformat="egrid")
    grd1 = xtgeo.grid_from_file(tmp_path / "E1.EGRID")
    assert grd1.dimensions == dimensions


def test_emerald_grid_values(emerald_grid):
    assert emerald_grid.name == "emerald_hetero_grid"
    assert emerald_grid.dimensions == (70, 100, 46)
    assert emerald_grid.nactive == 120842

    dzv = emerald_grid.get_dz()
    dzval = dzv.values
    mydz = float(dzval[31:32, 72:73, 0:1])
    assert mydz == pytest.approx(2.761, abs=0.001), "Grid DZ Emerald"
    dxv, dyv = (emerald_grid.get_dx(), emerald_grid.get_dy())

    mydx = float(dxv.values[31:32, 72:73, 0:1])
    mydy = float(dyv.values[31:32, 72:73, 0:1])

    assert mydx == pytest.approx(118.51, abs=0.01), "Grid DX Emerald"
    assert mydy == pytest.approx(141.21, abs=0.01), "Grid DY Emerald"

    xvv, yvv, zvv = emerald_grid.get_xyz(names=["xxx", "yyy", "zzz"])

    assert xvv.name == "xxx", "Name of X coord"
    xvv.name = "Xerxes"

    emerald_grid.props = [xvv, yvv]

    assert emerald_grid.get_prop_by_name("Xerxes").name == "Xerxes"


def test_roffbin_get_dataframe_for_grid(emerald_grid):
    df = emerald_grid.get_dataframe()

    assert len(df) == emerald_grid.nactive

    assert df["X_UTME"][0] == pytest.approx(459176.7937727844, abs=0.1)

    assert len(df.columns) == 6

    df = emerald_grid.get_dataframe(activeonly=False)
    assert len(df.columns) == 7
    assert len(df) != emerald_grid.nactive

    assert len(df) == np.prod(emerald_grid.dimensions)


def test_subgrids():
    grd = xtgeo.create_box_grid((10, 10, 46))

    newsub = {}
    newsub["XX1"] = 20
    newsub["XX2"] = 2
    newsub["XX3"] = 24

    grd.set_subgrids(newsub)
    assert grd.get_subgrids() == newsub

    _i_index, _j_index, k_index = grd.get_ijk()

    zprop = k_index.copy()
    zprop.values[k_index.values > 4] = 2
    zprop.values[k_index.values <= 4] = 1
    zprop.codes = {1: "Upper", 2: "Lower"}

    subgrids = grd.subgrids_from_zoneprop(zprop)
    assert "Upper" in subgrids

    calc_zprop = grd.get_zoneprop_from_subgrids()
    np.testing.assert_array_equal(zprop.values, calc_zprop.values)
    assert zprop.codes == calc_zprop.codes

    # rename
    grd.rename_subgrids(["AAAA", "BBBB"])
    assert "AAAA" in grd.subgrids

    # set to None
    grd.subgrids = None
    assert grd._subgrids is None


@functimer(output="info", comment="Roff binary import using new API, run 100 times")
def test_roffbin_import_v2stress(testdata_path):
    """Test roff binary import ROFF using new API, compare timing etc."""
    for _ino in range(100):
        xtgeo.grid_from_file(testdata_path / REEKFIL4)


def test_roff_coords_precision(testdata_path):
    """Secure that ROFF reads large coords (Y mostly) correct."""
    grd = xtgeo.grid_from_file(testdata_path / REEKFIL4)
    clist = grd.get_xyz_cell_corners(ijk=(41, 64, 1))

    assert clist[1] == pytest.approx(5933003.0812, abs=0.001)
    assert clist[4] == pytest.approx(5933036.8560, abs=0.001)


def test_get_xyz_cell_corners():
    """Test get_xyz_cell_corners."""
    grd = xtgeo.create_box_grid((10, 10, 10))
    grd_big = xtgeo.create_box_grid((100, 100, 100))

    # get corners for cell (2, 3, 4)
    corners = grd.get_xyz_cell_corners((2, 3, 4))

    assert len(corners) == 24
    assert corners[0] == pytest.approx(1.0, abs=0.001)
    assert corners[23] == pytest.approx(4.0, abs=0.001)

    # # get corners for all cells
    # allcorners = grd.get_xyz_corners()

    # assert len(allcorners) == grd.nactive * 24

    @functimer(output="print", comment="Iterate 100 times over get_xyz_cell_corners")
    def get_corners():
        for i in range(100):
            _ = grd_big.get_xyz_cell_corners((2, 3, 4))

    get_corners()


def test_convert_vs_xyz_cell_corners(testdata_path):
    grd1 = xtgeo.grid_from_file(testdata_path / BANAL6)
    grd2 = grd1.copy()

    assert grd1.get_xyz_cell_corners() == grd2.get_xyz_cell_corners()

    assert grd1.get_xyz_cell_corners((4, 2, 3)) == grd2.get_xyz_cell_corners((4, 2, 3))

    grd2._convert_xtgformat2to1()

    assert grd1.get_xyz_cell_corners((4, 2, 3)) == grd2.get_xyz_cell_corners((4, 2, 3))

    grd2._convert_xtgformat1to2()

    assert grd1.get_xyz_cell_corners((4, 2, 3)) == grd2.get_xyz_cell_corners((4, 2, 3))


def test_roff_bin_vs_ascii_export(tmp_path):
    grd1 = xtgeo.create_box_grid(dimension=(10, 10, 10))

    grd1.to_file(tmp_path / "b6_export.roffasc", fformat="roff_asc")
    grd1.to_file(tmp_path / "b6_export.roffbin", fformat="roff_bin")

    grd2 = xtgeo.grid_from_file(tmp_path / "b6_export.roffbin")
    cell1 = grd1.get_xyz_cell_corners((2, 2, 2))
    cell2 = grd2.get_xyz_cell_corners((2, 2, 2))

    assert cell1 == pytest.approx(cell2)


@pytest.mark.parametrize("xtgformat", [1, 2])
@pytest.mark.benchmark()
def test_benchmark_get_xyz_cell_cornerns(benchmark, xtgformat):
    grd = xtgeo.create_box_grid(dimension=(10, 10, 10))
    if xtgformat == 1:
        grd._set_xtgformat1()
    else:
        grd._set_xtgformat2()

    def run():
        return grd.get_xyz_cell_corners((5, 6, 7))

    corners = benchmark(run)

    assert corners == pytest.approx(
        [4, 5, 6, 5, 5, 6, 4, 6, 6, 5, 6, 6, 4, 5, 7, 5, 5, 7, 4, 6, 7, 5, 6, 7]
    )


def test_roffbin_import_wsubgrids(testdata_path):
    assert xtgeo.grid_from_file(testdata_path / REEKFIL5).subgrids == {
        "subgrid_0": range(1, 21),
        "subgrid_1": range(21, 41),
        "subgrid_2": range(41, 57),
    }


def test_import_grdecl_and_bgrdecl(testdata_path):
    """Eclipse import of GRDECL and binary GRDECL."""
    grd1 = xtgeo.grid_from_file(testdata_path / REEKFIL2, fformat="grdecl")
    grd2 = xtgeo.grid_from_file(testdata_path / REEKFIL3, fformat="bgrdecl")

    assert grd1.dimensions == (40, 64, 14)
    assert grd1.nactive == 35812

    assert grd2.dimensions == (40, 64, 14)
    assert grd2.nactive == 35812

    np.testing.assert_allclose(grd1.get_dz().values, grd2.get_dz().values, atol=0.001)


def test_eclgrid_import2(tmp_path, testdata_path):
    """Eclipse EGRID import, also change ACTNUM."""
    grd = xtgeo.grid_from_file(testdata_path / REEKFILE, fformat="egrid")

    assert grd.ncol == 40, "EGrid NX from Eclipse"
    assert grd.nrow == 64, "EGrid NY from Eclipse"
    assert grd.nactive == 35838, "EGrid NTOTAL from Eclipse"
    assert grd.ntotal == 35840, "EGrid NACTIVE from Eclipse"

    actnum = grd.get_actnum()
    assert actnum.values[12, 22, 5] == 0, "ACTNUM 0"

    actnum.values[:, :, :] = 1
    actnum.values[:, :, 4:6] = 0
    grd.set_actnum(actnum)
    newactive = grd.ncol * grd.nrow * grd.nlay - 2 * (grd.ncol * grd.nrow)
    assert grd.nactive == newactive, "Changed ACTNUM"
    grd.to_file(tmp_path / "reek_new_actnum.roff")


def test_eclgrid_import3(tmp_path, testdata_path):
    """Eclipse GRDECL import and translate."""
    grd = xtgeo.grid_from_file(testdata_path / BRILGRDECL, fformat="grdecl")

    mylist = grd.get_geometrics()

    xori1 = mylist[0]

    # translate the coordinates
    grd.translate_coordinates(translate=(100, 100, 10), flip=(1, 1, 1))

    mylist = grd.get_geometrics()

    xori2 = mylist[0]

    # check if origin is translated 100m in X
    assert xori1 + 100 == xori2, "Translate X distance"

    grd.to_file(tmp_path / "g1_translate.roff", fformat="roff_binary")

    grd.to_file(tmp_path / "g1_translate.bgrdecl", fformat="bgrdecl")


def test_geometrics_reek(testdata_path):
    """Import Reek and test geometrics."""
    grd = xtgeo.grid_from_file(testdata_path / REEKFILE, fformat="egrid")

    geom = grd.get_geometrics(return_dict=True, cellcenter=False)

    for key, val in geom.items():
        logger.info("%s is %s", key, val)

    # compared with RMS info:
    assert geom["xmin"] == pytest.approx(456510.6, abs=0.1), "Xmin"
    assert geom["ymax"] == pytest.approx(5938935.5, abs=0.1), "Ymax"

    # cellcenter True:
    geom = grd.get_geometrics(return_dict=True, cellcenter=True)
    assert geom["xmin"] == pytest.approx(456620, abs=1), "Xmin cell center"


def test_activate_all_cells(emerald_grid):
    emerald_grid.activate_all()
    assert emerald_grid.nactive == emerald_grid.ntotal


def test_get_adjacent_cells(tmp_path, emerald_grid):
    """Get the cell indices for discrete value X vs Y, if connected."""
    actnum = emerald_grid.get_actnum()
    actnum.to_file(tmp_path / "emerald_actnum.roff")
    result = emerald_grid.get_adjacent_cells(actnum, 0, 1, activeonly=False)
    assert result.name == "ADJ_CELLS"
    assert result.dimensions == emerald_grid.dimensions
    result.to_file(tmp_path / "emerald_adj_cells.roff")


def test_npvalues1d(testdata_path):
    """Different ways of getting np arrays."""
    grd = xtgeo.grid_from_file(testdata_path / DUALFIL3)
    dz = grd.get_dz()

    dz1 = dz.get_npvalues1d(activeonly=False)  # [  1.   1.   1.   1.   1.  nan  ...]
    dz2 = dz.get_npvalues1d(activeonly=True)  # [  1.   1.   1.   1.   1.  1. ...]

    assert dz1[0] == 1.0
    assert np.isnan(dz1[5])
    assert dz1[0] == 1.0
    assert not np.isnan(dz2[5])

    grd = xtgeo.grid_from_file(testdata_path / DUALFIL1)  # all cells active
    dz = grd.get_dz()

    dz1 = dz.get_npvalues1d(activeonly=False)
    dz2 = dz.get_npvalues1d(activeonly=True)

    assert dz1.all() == dz2.all()


def test_grid_design(emerald_grid):
    """Determine if a subgrid is topconform (T), baseconform (B), proportional (P).

    "design" refers to type of conformity
    "dzsimbox" is avg or representative simbox thickness per cell

    """
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


@pytest.mark.parametrize(
    "grid, flip",
    [
        (xtgeo.create_box_grid((30, 20, 3), flip=-1), -1),
        (xtgeo.create_box_grid((30, 20, 3), flip=1), 1),
        (xtgeo.create_box_grid((30, 20, 3), rotation=30, flip=1), 1),
        (xtgeo.create_box_grid((30, 20, 3), rotation=30, flip=-1), -1),
        (xtgeo.create_box_grid((30, 20, 3), rotation=190, flip=-1), -1),
        (xtgeo.create_box_grid((30, 20, 3), rotation=190, flip=1), 1),
    ],
)
def test_flip(grid, flip):
    """Determine if grid is flipped (lefthanded vs righthanded)."""
    assert grid.estimate_flip() == flip


def test_xyz_cell_corners(testdata_path):
    """Test xyz variations."""
    grd = xtgeo.grid_from_file(testdata_path / DUALFIL1)

    allcorners = grd.get_xyz_corners()
    assert len(allcorners) == 24
    assert allcorners[0].get_npvalues1d()[0] == 0.0
    assert allcorners[23].get_npvalues1d()[-1] == 1001.0


def test_grid_layer_slice(testdata_path):
    """Test grid slice coordinates."""
    grd = xtgeo.grid_from_file(testdata_path / REEKFILE)

    sarr1, _ibarr = grd.get_layer_slice(1)
    sarrn, _ibarr = grd.get_layer_slice(grd.nlay, top=False)

    cell1 = grd.get_xyz_cell_corners(ijk=(1, 1, 1))
    celln = grd.get_xyz_cell_corners(ijk=(1, 1, grd.nlay))
    celll = grd.get_xyz_cell_corners(ijk=(grd.ncol, grd.nrow, grd.nlay))

    assert sarr1[0, 0, 0] == cell1[0]
    assert sarr1[0, 0, 1] == cell1[1]

    assert sarrn[0, 0, 0] == celln[12]
    assert sarrn[0, 0, 1] == celln[13]

    assert sarrn[-1, 0, 0] == celll[12]
    assert sarrn[-1, 0, 1] == celll[13]


@given(xtgeo_grids)
def test_generate_hash(grd1):
    grd2 = grd1.copy()
    assert id(grd1) != id(grd2)
    assert grd1.generate_hash() == grd2.generate_hash()


def test_gridquality_properties(show_plot, testdata_path):
    """Get grid quality props."""
    grd1 = xtgeo.grid_from_file(testdata_path / GRIDQC1)

    props1 = grd1.get_gridquality_properties()
    minang = props1.get_prop_by_name("minangle_topbase")
    assert minang.values[5, 2, 1] == pytest.approx(71.05561, abs=0.001)
    if show_plot:
        lay = 1
        layslice = xtgeo.plot.Grid3DSlice()
        layslice.canvas(title=f"Layer {lay}")
        layslice.plot_gridslice(
            grd1,
            prop=minang,
            mode="layer",
            index=lay + 1,
            window=None,
            linecolor="black",
        )

        layslice.show()

    grd2 = xtgeo.grid_from_file(testdata_path / GRIDQC2)
    props2 = grd2.get_gridquality_properties()

    neg = props2.get_prop_by_name("negative_thickness")
    assert neg.values[0, 0, 0] == 0
    assert neg.values[2, 1, 0] == 1


def test_gridquality_properties_emerald(show_plot, emerald_grid):
    props = emerald_grid.get_gridquality_properties()

    concp = props.get_prop_by_name("concave_proj")
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


def test_bulkvol_banal6(testdata_path):
    """Test cell bulk volume calculation."""
    grd = xtgeo.grid_from_file(testdata_path / BANAL6)
    bulkvol = grd.get_bulk_volume()

    bulkvol_sum = np.sum(bulkvol.values)

    assert bulkvol_sum == pytest.approx(29921.874972059588, abs=0.001)


def test_bulkvol(testdata_path):
    """Test cell bulk volume calculation."""
    grd = xtgeo.grid_from_file(testdata_path / GRIDQC1)
    cellvol_rms = xtgeo.gridproperty_from_file(testdata_path / GRIDQC1_CELLVOL)
    bulkvol = grd.get_bulk_volume()

    assert grd.dimensions == bulkvol.dimensions
    assert np.allclose(cellvol_rms.values, bulkvol.values)


@pytest.mark.benchmark(group="bulkvol")
def test_benchmark_bulkvol(benchmark):
    dimens = (10, 50, 5)
    grd = xtgeo.create_box_grid(dimension=dimens)

    def run():
        _ = grd.get_bulk_volume()

    benchmark(run)


def test_phasevol(testdata_path):
    """Test cell phase volume calculation."""
    grd = xtgeo.grid_from_file(testdata_path / GRIDQC1)
    cellvol_rms = xtgeo.gridproperty_from_file(testdata_path / GRIDQC1_CELLVOL)

    gasvol, oilvol, watervol = grd.get_phase_volume(woc=2000, goc=1800, boundary=None)

    assert grd.dimensions == gasvol.dimensions
    assert grd.dimensions == oilvol.dimensions
    assert grd.dimensions == watervol.dimensions
    bulkvol = gasvol.values + oilvol.values + watervol.values

    assert np.sum(gasvol.values) == pytest.approx(297890391)
    assert np.sum(oilvol.values) == pytest.approx(15132726)
    assert np.sum(watervol.values) == pytest.approx(86876861)

    assert np.allclose(cellvol_rms.values, bulkvol)


def test_phase_volume_with_polygon_boundary(testdata_path):
    """Test phase volume calculation with a polygon boundary constraint."""
    grid = xtgeo.create_box_grid(
        (10, 10, 5), origin=(0, 0, 1000), increment=(100, 100, 10)
    )

    # Create a simple rectangular polygon
    polygon_coords = np.array(
        [
            [200, 200, 1000],
            [800, 200, 1000],
            [800, 800, 1000],
            [200, 800, 1000],
            [200, 200, 1000],  # Close the polygon
        ]
    )
    poly = xtgeo.Polygons(polygon_coords)
    goc = xtgeo.GridProperty(grid, values=1020.0)
    woc = xtgeo.GridProperty(grid, values=1040.0)

    # Calculate with boundary
    gas_poly, oil_poly, water_poly = grid.get_phase_volume(
        woc=woc, goc=goc, boundary=poly
    )

    # Calculate without boundary
    gas_full, oil_full, water_full = grid.get_phase_volume(
        woc=woc, goc=goc, boundary=None
    )

    # Volumes with boundary should be less than or equal to full volumes
    assert np.sum(gas_poly.values) < np.sum(gas_full.values)
    assert np.sum(oil_poly.values) < np.sum(oil_full.values)
    assert np.sum(water_poly.values) < np.sum(water_full.values)


@pytest.mark.benchmark(group="phase_volume")
def test_benchmark_phase_volume(benchmark):
    """Benchmark phase volume calculation."""
    grid = xtgeo.create_box_grid(
        (10, 50, 5), origin=(0, 0, 1000), increment=(100, 100, 10)
    )
    goc = xtgeo.GridProperty(grid, values=1030.0)
    woc = xtgeo.GridProperty(grid, values=1060.0)

    def run():
        _ = grid.get_phase_volume(woc=woc, goc=goc, boundary=None)

    benchmark(run)


@pytest.mark.benchmark(group="phase_volume")
def test_benchmark_phase_volume_with_boundary(benchmark):
    """Benchmark phase volume calculation with boundary."""
    grid = xtgeo.create_box_grid(
        (10, 50, 5), origin=(0, 0, 1000), increment=(100, 100, 10)
    )
    goc = xtgeo.GridProperty(grid, values=1015.0)
    woc = xtgeo.GridProperty(grid, values=1040.0)
    polygon_coords = np.array(
        [
            [300, 1000, 1000],
            [600, 1000, 1000],
            [600, 3000, 1000],
            [300, 3000, 1000],
            [300, 1000, 1000],
        ]
    )

    def run():
        _ = grid.get_phase_volume(
            woc=woc, goc=goc, boundary=xtgeo.Polygons(polygon_coords)
        )

    benchmark(run)


def test_cell_height_above_ffl(testdata_path):
    """Test cell heights above ffl."""
    grd = xtgeo.grid_from_file(testdata_path / GRIDQC1)

    ffl = xtgeo.GridProperty(grd, values=1700)

    htop, hbot, hmid = grd.get_heights_above_ffl(ffl, option="cell_center_above_ffl")

    assert htop.values[6, 4, 0] == pytest.approx(65.8007)
    assert hbot.values[6, 4, 0] == pytest.approx(54.1729)
    assert hmid.values[6, 4, 0] == pytest.approx(59.9868)
    assert hbot.values[4, 0, 0] == pytest.approx(0.0)

    htop, hbot, hmid = grd.get_heights_above_ffl(ffl, option="cell_corners_above_ffl")

    assert htop.values[4, 5, 0] == pytest.approx(44.8110)
    assert hbot.values[4, 5, 0] == pytest.approx(0.0)
    assert hmid.values[4, 5, 0] == pytest.approx(22.4055)

    htop, hbot, hmid = grd.get_heights_above_ffl(
        ffl, option="truncated_cell_corners_above_ffl"
    )

    assert htop.values[4, 5, 0] == pytest.approx(11.202758)
    assert hbot.values[4, 5, 0] == pytest.approx(8.35742)
    assert hmid.values[4, 5, 0] == pytest.approx(9.78009)


@functimer(output="info")
def test_get_property_between_surfaces(testdata_path):
    """Generate a marker property between two surfaces."""
    grd = xtgeo.grid_from_file(testdata_path / REEKFIL4)

    surf1 = xtgeo.surface_from_grid3d(grd)
    surf1.fill()
    surf1.values = 1650
    surf2 = surf1.copy()
    surf2.values = 1700

    prop = grd.get_property_between_surfaces(surf1, surf2)

    assert prop.values.sum() == 137269  # verified with similar method in RMS

    # multiply values with 2
    prop2 = grd.get_property_between_surfaces(surf1, surf2, value=2)
    assert prop2.values.sum() == 137269 * 2

    # swap one if the surfaces so yflip becomes -1
    surf1.make_righthanded()
    prop2 = grd.get_property_between_surfaces(surf1, surf2)
    assert prop2.values.sum() == 137269


def test_get_property_between_surfaces_w_holes(testdata_path):
    """Generate a marker property between two surfaces, where surfaces has holes."""
    grd = xtgeo.grid_from_file(testdata_path / REEKFIL4)

    surf1 = xtgeo.surface_from_grid3d(grd)
    surf1.fill()
    surf1.values = 1650
    surf2 = surf1.copy()
    surf2.values = 1700

    surf1.values.mask[60:70, 70:75] = True
    surf2.values.mask[50:70, 60:71] = True

    prop = grd.get_property_between_surfaces(surf1, surf2)

    assert prop.values.sum() == 129592  # ~verified  in RMS (129591 there ~precision)


def test_bad_egrid_ends_before_kw(tmp_path):
    egrid_file = tmp_path / "test.egrid"
    with open(egrid_file, "wb") as fh:
        fh.write(b"\x00\x00\x00\x10")
    with pytest.raises(Exception, match="end-of-file while reading keyword"):
        xtgeo.grid_from_file(egrid_file, fformat="egrid")


@given(dimensions, increments, increments, increments)
def test_grid_get_dx(dimension, dx, dy, dz):
    grd = xtgeo.create_box_grid(
        dimension=dimension, increment=(dx, dy, dz), rotation=0.0
    )
    np.testing.assert_allclose(grd.get_dx(metric="euclid").values, dx, atol=0.01)
    np.testing.assert_allclose(
        grd.get_dx(metric="north south vertical").values, 0.0, atol=0.01
    )
    np.testing.assert_allclose(
        grd.get_dx(metric="east west vertical").values, dx, atol=0.01
    )
    np.testing.assert_allclose(grd.get_dx(metric="horizontal").values, dx, atol=0.01)
    np.testing.assert_allclose(grd.get_dx(metric="x projection").values, dx, atol=0.01)
    np.testing.assert_allclose(grd.get_dx(metric="y projection").values, 0.0, atol=0.01)
    np.testing.assert_allclose(grd.get_dx(metric="z projection").values, 0.0, atol=0.01)

    grd._actnumsv[0, 0, 0] = 0

    assert grd.get_dx(asmasked=True).values[0, 0, 0] is np.ma.masked
    assert np.isclose(grd.get_dx(asmasked=False).values[0, 0, 0], dx, atol=0.01)


@given(dimensions, increments, increments, increments)
def test_grid_get_dy(dimension, dx, dy, dz):
    grd = xtgeo.create_box_grid(
        dimension=dimension, increment=(dx, dy, dz), rotation=0.0
    )
    np.testing.assert_allclose(grd.get_dy(metric="euclid").values, dy, atol=0.01)
    np.testing.assert_allclose(
        grd.get_dy(metric="north south vertical").values, dy, atol=0.01
    )
    np.testing.assert_allclose(
        grd.get_dy(metric="east west vertical").values, 0.0, atol=0.01
    )
    np.testing.assert_allclose(grd.get_dy(metric="horizontal").values, dy, atol=0.01)
    np.testing.assert_allclose(grd.get_dy(metric="x projection").values, 0.0, atol=0.01)
    np.testing.assert_allclose(grd.get_dy(metric="y projection").values, dy, atol=0.01)
    np.testing.assert_allclose(grd.get_dy(metric="z projection").values, 0.0, atol=0.01)

    grd._actnumsv[0, 0, 0] = 0

    assert grd.get_dy(asmasked=True).values[0, 0, 0] is np.ma.masked
    assert np.isclose(grd.get_dy(asmasked=False).values[0, 0, 0], dy, atol=0.01)


@given(dimensions, increments, increments, increments)
def test_grid_get_dz(dimension, dx, dy, dz):
    grd = xtgeo.create_box_grid(dimension=dimension, increment=(dx, dy, dz))
    np.testing.assert_allclose(grd.get_dz(metric="euclid").values, dz, atol=0.01)
    np.testing.assert_allclose(
        grd.get_dz(metric="north south vertical").values, dz, atol=0.01
    )
    np.testing.assert_allclose(
        grd.get_dz(metric="east west vertical").values, dz, atol=0.01
    )
    np.testing.assert_allclose(grd.get_dz(metric="horizontal").values, 0.0, atol=0.01)
    np.testing.assert_allclose(grd.get_dz(metric="x projection").values, 0.0, atol=0.01)
    np.testing.assert_allclose(grd.get_dz(metric="y projection").values, 0.0, atol=0.01)
    np.testing.assert_allclose(grd.get_dz(metric="z projection").values, dz, atol=0.01)
    np.testing.assert_allclose(grd.get_dz(flip=False).values, -dz, atol=0.01)

    grd._actnumsv[0, 0, 0] = 0

    assert grd.get_dz(asmasked=True).values[0, 0, 0] is np.ma.masked
    assert np.isclose(grd.get_dz(asmasked=False).values[0, 0, 0], dz, atol=0.01)


def test_benchmark_grid_get_dz(benchmark):
    grd = xtgeo.create_box_grid(dimension=(100, 100, 100))

    def run():
        grd.get_dz()

    benchmark(run)


def test_grid_get_dxdydz_zero_size():
    grd = xtgeo.create_box_grid(dimension=(0, 0, 0))

    assert grd.get_dx().values.shape == (0, 0, 0)
    assert grd.get_dy().values.shape == (0, 0, 0)
    assert grd.get_dz().values.shape == (0, 0, 0)


# def test_grid_get_dxdydz_bad_coordsv_size():
#     grd = xtgeo.create_box_grid(dimension=(10, 10, 10))
#     grd._coordsv = np.zeros(shape=(0, 0, 0))

#     with pytest.raises(RuntimeError, match="should"):
#         grd.get_dx()
#     with pytest.raises(RuntimeError, match="should"):
#         grd.get_dy()
#     with pytest.raises(xtgeo.XTGeoCLibError, match="Incorrect size of coordsv"):
#         grd.get_dz()


# def test_grid_get_dxdydz_bad_zcorn_size():
#     grd = xtgeo.create_box_grid(dimension=(10, 10, 10))
#     grd._zcornsv = np.zeros(shape=(0, 0, 0, 0))

#     with pytest.raises(xtgeo.XTGeoCLibError, match="Incorrect size of zcornsv"):
#         grd.get_dx()
#     with pytest.raises(xtgeo.XTGeoCLibError, match="Incorrect size of zcornsv"):
#         grd.get_dy()
#     with pytest.raises(xtgeo.XTGeoCLibError, match="Incorrect size of zcornsv"):
#         grd.get_dz()


# def test_grid_get_dxdydz_bad_grid_top():
#     grd = xtgeo.create_box_grid(dimension=(10, 10, 10))

#     grd._coordsv[:, :, 2] = 0.0
#     grd._coordsv[:, :, 5] = 0.0
#     grd._coordsv[:, :, 0] += 1.0

#     with pytest.raises(xtgeo.XTGeoCLibError, match="has near zero height"):
#         grd.get_dx()
#     with pytest.raises(xtgeo.XTGeoCLibError, match="has near zero height"):
#         grd.get_dy()
#     with pytest.raises(xtgeo.XTGeoCLibError, match="has near zero height"):
#         grd.get_dz()


# def test_grid_get_dxdydz_bad_metric():
#     grd = xtgeo.create_box_grid(dimension=(10, 10, 10))

#     with pytest.raises(ValueError, match="Unknown metric"):
#         grd.get_dx(metric="foo")
#     with pytest.raises(ValueError, match="Unknown metric"):
#         grd.get_dy(metric="foo")
#     with pytest.raises(ValueError, match="Unknown metric"):
#         grd.get_dz(metric="foo")


def test_grid_roff_subgrids_import_regression(tmp_path):
    grid = xtgeo.create_box_grid(dimension=(5, 5, 67))
    grid.subgrids = {
        "subgrid_0": list(range(1, 21)),
        "subgrid_1": list(range(21, 53)),
        "subgrid_2": list(range(53, 68)),
    }
    grid.to_file(tmp_path / "grid.roff")

    grid2 = xtgeo.grid_from_file(tmp_path / "grid.roff")
    assert grid2.subgrids == {
        "subgrid_0": range(1, 21),
        "subgrid_1": range(21, 53),
        "subgrid_2": range(53, 68),
    }


@pytest.mark.parametrize(
    "coordsv_dtype, zcornsv_dtype, actnumsv_dtype, match",
    [
        (np.float32, np.float32, np.int32, "The dtype of the coordsv"),
        (np.float64, np.float64, np.int32, "The dtype of the zcornsv"),
        (np.float64, np.float32, np.uint8, "The dtype of the actnumsv"),
    ],
)
def test_grid_bad_dtype_construction(
    coordsv_dtype, zcornsv_dtype, actnumsv_dtype, match
):
    with pytest.raises(TypeError, match=match):
        Grid(
            np.zeros((2, 2, 6), dtype=coordsv_dtype),
            np.zeros((2, 2, 2, 4), dtype=zcornsv_dtype),
            np.zeros((1, 1, 1), dtype=actnumsv_dtype),
        )


@pytest.mark.parametrize(
    "coordsv_dimensions, zcornsv_dimensions, actnumsv_dimensions, match",
    [
        ((2, 2, 2), (2, 2, 2, 4), (1, 1, 1), "shape of coordsv"),
        ((2, 2, 6), (2, 2, 2, 3), (1, 1, 1), "shape of zcornsv"),
        ((2, 2, 6), (2, 1, 2, 4), (1, 1, 1), "Mismatch between zcornsv and coordsv"),
        ((2, 2, 6), (2, 2, 2, 4), (1, 2, 1), "Mismatch between zcornsv and actnumsv"),
    ],
)
def test_grid_bad_dimensions_construction(
    coordsv_dimensions, zcornsv_dimensions, actnumsv_dimensions, match
):
    with pytest.raises(ValueError, match=match):
        Grid(
            np.zeros(coordsv_dimensions, dtype=np.float64),
            np.zeros(zcornsv_dimensions, dtype=np.float32),
            np.zeros(actnumsv_dimensions, dtype=np.int32),
        )


def test_get_vtk_geometries_box(show_plot):
    grd = xtgeo.create_box_grid((2, 6, 4), increment=(20, 10, 7))
    grd._actnumsv[0, 0, 0] = 0
    grd._actnumsv[1, 0, 0] = 0
    grd._actnumsv[1, 2, 0] = 0
    grd._actnumsv[0, 0, 1] = 0

    dims, corners, indi = grd.get_vtk_geometries()

    assert list(dims) == [3, 7, 5]
    assert list(indi) == [0, 1, 5, 12]
    assert list(corners.flatten()[0:5]) == [0.0, 0.0, 0.0, 20.0, 0.0]

    if show_plot:
        try:
            import pyvista as pv  # type: ignore
        except ImportError:
            warnings.warn("show_plot is active but no pyvista installed")
            return
        grid = pv.ExplicitStructuredGrid(dims, corners)
        grid = grid.compute_connectivity()
        grid.flip_z(inplace=True)

        grid = grid.hide_cells(indi)
        grid.plot(show_edges=True)


def test_get_vtk_geometries_emerald(show_plot, testdata_path):
    grd = xtgeo.grid_from_file(testdata_path / EME1)

    dims, corners, indi = grd.get_vtk_geometries()
    assert corners.mean() == pytest.approx(2132426.94, abs=0.01)

    poro = xtgeo.gridproperty_from_file(testdata_path / EME1PROP, name="PORO")
    porov = poro.get_npvalues1d(order="F")

    if show_plot:
        try:
            import pyvista as pv  # type: ignore
        except ImportError:
            warnings.warn("show_plot is active but no pyvista installed")
            return
        grid = pv.ExplicitStructuredGrid(dims, corners)
        grid.flip_z(inplace=True)
        grid = grid.hide_cells(indi)
        grid.cell_data["PORO"] = porov
        pv.global_theme.show_edges = True
        plt = pv.Plotter()
        plt.add_mesh(grid, clim=[0, 0.4])
        plt.set_scale(1, 1, 5)
        plt.show_axes()
        plt.show()


def test_get_vtk_esg_geometry_data_four_cells():
    """Test that extracted VTK ESG connectivity and vertex coordinates are correct
    for simple grid with only four cells in a single layer"""
    grd = xtgeo.create_box_grid((2, 2, 1), increment=(1, 3, 10))

    # Note that returned dims is in terms of points
    dims, vertex_arr, conn_arr, inactive_cell_indices = grd.get_vtk_esg_geometry_data()

    assert list(dims) == [3, 3, 2]
    assert len(vertex_arr) == 3 * 3 * 2
    assert len(conn_arr) == 8 * 4
    assert len(inactive_cell_indices) == 0

    # Bounding box of first cell should be min/max x=[0,1] y=[0,3], z[0,10]
    assert vertex_arr[conn_arr[0]] == pytest.approx([0, 0, 0])
    assert vertex_arr[conn_arr[1]] == pytest.approx([1, 0, 0])
    assert vertex_arr[conn_arr[2]] == pytest.approx([1, 3, 0])
    assert vertex_arr[conn_arr[3]] == pytest.approx([0, 3, 0])
    assert vertex_arr[conn_arr[4]] == pytest.approx([0, 0, 10])
    assert vertex_arr[conn_arr[5]] == pytest.approx([1, 0, 10])
    assert vertex_arr[conn_arr[6]] == pytest.approx([1, 3, 10])
    assert vertex_arr[conn_arr[7]] == pytest.approx([0, 3, 10])

    # Bounding box of fourth cell should be min/max x=[1,2] y=[3,6], z[0,10]
    assert vertex_arr[conn_arr[24]] == pytest.approx([1, 3, 0])
    assert vertex_arr[conn_arr[25]] == pytest.approx([2, 3, 0])
    assert vertex_arr[conn_arr[26]] == pytest.approx([2, 6, 0])
    assert vertex_arr[conn_arr[27]] == pytest.approx([1, 6, 0])
    assert vertex_arr[conn_arr[28]] == pytest.approx([1, 3, 10])
    assert vertex_arr[conn_arr[29]] == pytest.approx([2, 3, 10])
    assert vertex_arr[conn_arr[30]] == pytest.approx([2, 6, 10])
    assert vertex_arr[conn_arr[31]] == pytest.approx([1, 6, 10])


def test_get_vtk_esg_geometry_data_box():
    """Test that extracted VTK ESG geometry looks sensible for small test grid with
    incative cells"""
    grd = xtgeo.create_box_grid((2, 6, 4), increment=(20, 10, 7))
    grd._actnumsv[0, 0, 0] = 0
    grd._actnumsv[1, 0, 0] = 0

    dims, vertex_arr, conn_arr, inactive_cell_indices = grd.get_vtk_esg_geometry_data()

    assert list(dims) == [3, 7, 5]
    assert len(vertex_arr) == 3 * 7 * 5
    assert len(conn_arr) == 8 * 2 * 6 * 4
    assert len(inactive_cell_indices) == 2


def test_grid_cache():
    """Test internal grid cache, which relates to class _GridCache."""
    grd1 = xtgeo.create_box_grid(dimension=(10, 10, 10))
    cache1 = grd1._get_cache()

    grd2 = xtgeo.create_box_grid(dimension=(10, 10, 10))  # identical, but other id
    cache2 = grd2._get_cache()

    # they will get identical hashes, but different id on hashes
    assert cache1.hash == cache2.hash
    assert id(cache1) != id(cache2)

    # update of cache1 should receive same id

    cache1_updated = grd1._get_cache()

    assert id(cache1_updated) == id(cache1)

    # but changing the grid should change the cache id
    grd1._zcornsv += 0.001
    cache1_updated = grd1._get_cache()
    assert id(cache1_updated) != id(cache1)

    # if the grid is copied, the cache id may change but cache hash should be the same
    grd1_copy = grd1.copy()
    cache1_copy = grd1_copy._get_cache()
    assert id(cache1_copy) != id(cache1_updated)
    assert cache1_copy.hash == cache1_updated.hash


def test_collapse_inactive_cells(testdata_path):
    """Test collapsing of inactive cells."""
    grd = xtgeo.grid_from_file(testdata_path / B9)
    grd._actnumsv[1, 0, 0:2] = 0
    grd._actnumsv[1, 0, 3] = 0
    grd._actnumsv[2, 0, 4:6] = 0
    grd._actnumsv[2, 0, 2] = 0
    grd._actnumsv[3, 1, :] = 0

    assert grd._zcornsv[1, 0, 0, 3] == pytest.approx(0.0)
    assert grd._zcornsv[1, 0, 6, 3] == pytest.approx(3.0)

    g1 = grd.copy()
    g2 = grd.copy()

    g1.collapse_inactive_cells(internal=True)
    assert g1._zcornsv[1, 0, 0, 3] == pytest.approx(1.25)
    assert g1._zcornsv[1, 0, 6, 3] == pytest.approx(3.0)
    assert g1._zcornsv[1, 0, 4, 3] == pytest.approx(1.8125)
    assert g1._zcornsv[2, 1, 4, 0] == pytest.approx(2.0)

    g2.collapse_inactive_cells(internal=False)
    assert g2._zcornsv[1, 0, 0, 3] == pytest.approx(1.25)
    assert g2._zcornsv[1, 0, 6, 3] == pytest.approx(3.0)
    assert g2._zcornsv[1, 0, 4, 3] == pytest.approx(2.0)
    assert g2._zcornsv[2, 1, 4, 0] == pytest.approx(2.25)


def test_more_translate_coords(testdata_path):
    """Extended tests for translate_coordinates functionality."""
    g = xtgeo.grid_from_file(testdata_path / B7)
    disc = xtgeo.gridproperty_from_file(testdata_path / B7, name="DISC")
    cont = xtgeo.gridproperty_from_file(testdata_path / B7, name="CONT")

    g.props = [disc, cont]

    assert g.dimensions == (4, 2, 3)

    assert g.get_subgrids() == {"subgrid_0": 1, "subgrid_1": 2}

    g.rename_subgrids(["link", "zelda"])
    assert g.get_subgrids() == {"link": 1, "zelda": 2}

    # Store original values for comparison
    original_xyz = g.get_xyz()
    original_x_mean = original_xyz[0].values.mean()
    original_y_mean = original_xyz[1].values.mean()
    original_z_mean = original_xyz[2].values.mean()
    original_handedness = g.ijk_handedness

    # Test 1: Simple translation
    g1 = g.copy()
    g1.translate_coordinates(translate=(100.0, 200.0, 50.0))

    new_xyz = g1.get_xyz()
    assert new_xyz[0].values.mean() == pytest.approx(original_x_mean + 100.0, abs=0.001)
    assert new_xyz[1].values.mean() == pytest.approx(original_y_mean + 200.0, abs=0.001)
    assert new_xyz[2].values.mean() == pytest.approx(original_z_mean + 50.0, abs=0.001)

    # Properties should be preserved
    assert len(g1.props) == 2
    assert g1.props[0].name == "DISC"
    assert g1.props[1].name == "CONT"

    # Test 2: Rotation
    g2 = g.copy()
    g2.translate_coordinates(add_rotation=45.0)

    # After rotation, coordinates should be different
    rotated_xyz = g2.get_xyz()
    assert not np.allclose(rotated_xyz[0].values, original_xyz[0].values)
    assert not np.allclose(rotated_xyz[1].values, original_xyz[1].values)
    # Z should be unchanged for rotation
    np.testing.assert_allclose(rotated_xyz[2].values, original_xyz[2].values)

    # Test 3: Flipping axes
    g3 = g.copy()
    g3.translate_coordinates(flip=(-1, 1, 1))  # Flip X axis

    assert g3.ijk_handedness != original_handedness

    # Test 4: Vertical flip
    g4 = g.copy()
    g4.translate_coordinates(flip=(1, 1, -1))  # Flip Z axis
    assert g4.get_subgrids() == {"zelda": 2, "link": 1}
    assert g4.subgrids == {"zelda": range(1, 3), "link": range(3, 4)}

    vflipped_xyz = g4.get_xyz()
    assert g4.ijk_handedness != original_handedness
    # Z values should be different after vertical flip
    assert not np.allclose(vflipped_xyz[2].values, original_xyz[2].values)

    # Test 5: Row axis flip
    g5 = g.copy()
    g5.translate_coordinates(flip=(1, -1, 1))  # Flip Y axis (rows)

    # Test 6: Target coordinates
    g6 = g.copy()
    target = (1000.0, 2000.0, 3000.0)
    g6.translate_coordinates(target_coordinates=target)

    target_xyz = g6.get_xyz()
    assert target_xyz[0].values.mean() == pytest.approx(target[0], abs=0.001)
    assert target_xyz[1].values.mean() == pytest.approx(target[1], abs=0.001)
    assert target_xyz[2].values.mean() == pytest.approx(target[2], abs=0.001)

    # Test 7: Combined operations (rotation + translation + flip)
    g7 = g.copy()
    g7.translate_coordinates(
        add_rotation=30.0, translate=(50.0, 100.0, 25.0), flip=(1, 1, -1)
    )

    combined_xyz = g7.get_xyz()
    # Should be significantly different from original
    assert not np.allclose(combined_xyz[0].values, original_xyz[0].values)
    assert not np.allclose(combined_xyz[1].values, original_xyz[1].values)
    assert not np.allclose(combined_xyz[2].values, original_xyz[2].values)

    # Test 8: Error case - both translate and target_coordinates
    g8 = g.copy()
    with pytest.raises(
        ValueError, match="Using both key 'translate' and key 'target_coordinates'"
    ):
        g8.translate_coordinates(
            translate=(10.0, 20.0, 30.0), target_coordinates=(100.0, 200.0, 300.0)
        )

    # Test 9: Zero translation should not change anything
    g9 = g.copy()
    g9.translate_coordinates(translate=(0.0, 0.0, 0.0))

    unchanged_xyz = g9.get_xyz()
    np.testing.assert_allclose(unchanged_xyz[0].values, original_xyz[0].values)
    np.testing.assert_allclose(unchanged_xyz[1].values, original_xyz[1].values)
    np.testing.assert_allclose(unchanged_xyz[2].values, original_xyz[2].values)

    # Test 10: Rotation with custom rotation point
    g10 = g.copy()
    rotation_point = (original_x_mean, original_y_mean)
    g10.translate_coordinates(add_rotation=90.0, rotation_point=rotation_point)

    rotated_custom_xyz = g10.get_xyz()
    # Should rotate around the specified point, not the default corner
    assert not np.allclose(rotated_custom_xyz[0].values, original_xyz[0].values)
    assert not np.allclose(rotated_custom_xyz[1].values, original_xyz[1].values)

    # Test 11: Go through all double or 360 degrees --> same result
    g11 = g.copy()
    g11.translate_coordinates(
        add_rotation=360.0, flip=(-1, -1, -1), translate=(10, 30, 20)
    )
    g11.translate_coordinates(
        add_rotation=360.0, flip=(-1, -1, -1), translate=(-10, -30, -20)
    )

    g11_xyz = g11.get_xyz()
    np.testing.assert_allclose(
        g11_xyz[0].values, original_xyz[0].values, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        g11_xyz[1].values, original_xyz[1].values, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        g11_xyz[2].values, original_xyz[2].values, rtol=1e-5, atol=1e-6
    )
