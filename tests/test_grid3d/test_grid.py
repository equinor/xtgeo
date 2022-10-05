"""Tests for 3D grid."""
import math
import warnings
from collections import OrderedDict
from os.path import join

import numpy as np
import pytest
import xtgeo
from hypothesis import given
from xtgeo.common import XTGeoDialog
from xtgeo.grid3d import Grid

from .grid_generator import dimensions, increments, xtgeo_grids

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__, info=True)

if not xtg.testsetup():
    raise SystemExit

TPATH = xtg.testpathobj

REEKFILE = TPATH / "3dgrids/reek/REEK.EGRID"
REEKFIL2 = TPATH / "3dgrids/reek3/reek_sim.grdecl"  # ASCII GRDECL
REEKFIL3 = TPATH / "3dgrids/reek3/reek_sim.bgrdecl"  # binary GRDECL
REEKFIL4 = TPATH / "3dgrids/reek/reek_geo_grid.roff"
REEKFIL5 = TPATH / "3dgrids/reek/reek_geo2_grid_3props.roff"
# brilfile = '../xtgeo-testdata/3dgrids/bri/B.GRID' ...disabled
BRILGRDECL = TPATH / "3dgrids/bri/b.grdecl"
BANAL6 = TPATH / "3dgrids/etc/banal6.roff"
GRIDQC1 = TPATH / "3dgrids/etc/gridqc1.roff"
GRIDQC1_CELLVOL = TPATH / "3dgrids/etc/gridqc1_totbulk.roff"
GRIDQC2 = TPATH / "3dgrids/etc/gridqc_negthick_twisted.roff"

DUALFIL1 = TPATH / "3dgrids/etc/dual_grid.roff"
DUALFIL3 = TPATH / "3dgrids/etc/TEST_DPDK.EGRID"

EME1 = TPATH / "3dgrids/eme/2/eme_small_w_hole_grid_params.roff"
EME1PROP = TPATH / "3dgrids/eme/2/eme_small_w_hole_grid_params.roff"

# =============================================================================
# Do tests
# =============================================================================
# pylint: disable=redefined-outer-name


@pytest.fixture()
def emerald_grid(testpath):
    return xtgeo.grid_from_file(
        join(testpath, "3dgrids/eme/1/emerald_hetero_grid.roff")
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

    with pytest.warns(DeprecationWarning, match="create_box is deprecated"):
        grd = Grid()
        grd.create_box((4, 3, 5))
        grd.to_file(tmp_path / "shoebox_default2.roff")

    grd = xtgeo.create_box_grid((2, 3, 4), flip=-1)
    grd.to_file(tmp_path / "shoebox_default_flipped.roff")

    timer1 = xtg.timer()
    grd = xtgeo.create_box_grid(
        origin=(0, 0, 1000), dimension=(300, 200, 30), increment=(20, 20, 1), flip=-1
    )
    logger.info("Making a a 1,8 mill cell grid took %5.3f secs", xtg.timer(timer1))

    dx, dy = grd.get_dxdy()

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
    dxv, dyv = emerald_grid.get_dxdy()

    mydx = float(dxv.values3d[31:32, 72:73, 0:1])
    mydy = float(dyv.values3d[31:32, 72:73, 0:1])

    assert mydx == pytest.approx(118.51, abs=0.01), "Grid DX Emerald"
    assert mydy == pytest.approx(141.26, abs=0.01), "Grid DY Emerald"

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

    newsub = OrderedDict()
    newsub["XX1"] = 20
    newsub["XX2"] = 2
    newsub["XX3"] = 24

    grd.set_subgrids(newsub)
    assert grd.get_subgrids() == newsub

    _i_index, _j_index, k_index = grd.get_ijk()

    zprop = k_index.copy()
    zprop.values[k_index.values > 4] = 2
    zprop.values[k_index.values <= 4] = 1

    grd.subgrids_from_zoneprop(zprop)

    # rename
    grd.rename_subgrids(["AAAA", "BBBB"])
    assert "AAAA" in grd.subgrids.keys()

    # set to None
    grd.subgrids = None
    assert grd._subgrids is None


def test_roffbin_import_v2stress():
    """Test roff binary import ROFF using new API, comapre timing etc."""
    t0 = xtg.timer()
    for _ino in range(100):
        xtgeo.grid_from_file(REEKFIL4)
    t1 = xtg.timer(t0)
    print("100 loops with ROXAPIV 2 took: ", t1)


def test_convert_vs_xyz_cell_corners():
    grd1 = xtgeo.grid_from_file(BANAL6)
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
        grd._xtgformat1()
    else:
        grd._xtgformat2()

    def run():
        return grd.get_xyz_cell_corners((5, 6, 7))

    corners = benchmark(run)

    assert corners == pytest.approx(
        [4, 5, 6, 5, 5, 6, 4, 6, 6, 5, 6, 6, 4, 5, 7, 5, 5, 7, 4, 6, 7, 5, 6, 7]
    )


def test_roffbin_import_wsubgrids():
    assert xtgeo.grid_from_file(REEKFIL5).subgrids == OrderedDict(
        [
            ("subgrid_0", range(1, 21)),
            ("subgrid_1", range(21, 41)),
            ("subgrid_2", range(41, 57)),
        ]
    )


def test_import_grdecl_and_bgrdecl():
    """Eclipse import of GRDECL and binary GRDECL."""
    grd1 = xtgeo.grid_from_file(REEKFIL2, fformat="grdecl")
    grd2 = xtgeo.grid_from_file(REEKFIL3, fformat="bgrdecl")

    assert grd1.dimensions == (40, 64, 14)
    assert grd1.nactive == 35812

    assert grd2.dimensions == (40, 64, 14)
    assert grd2.nactive == 35812

    np.testing.assert_allclose(grd1.get_dz().values, grd2.get_dz().values, atol=0.001)


def test_eclgrid_import2(tmp_path):
    """Eclipse EGRID import, also change ACTNUM."""
    grd = xtgeo.grid_from_file(REEKFILE, fformat="egrid")

    assert grd.ncol == 40, "EGrid NX from Eclipse"
    assert grd.nrow == 64, "EGrid NY from Eclipse"
    assert grd.nactive == 35838, "EGrid NTOTAL from Eclipse"
    assert grd.ntotal == 35840, "EGrid NACTIVE from Eclipse"

    actnum = grd.get_actnum()
    print(actnum.values[12:13, 22:24, 5:6])
    assert actnum.values[12, 22, 5] == 0, "ACTNUM 0"

    actnum.values[:, :, :] = 1
    actnum.values[:, :, 4:6] = 0
    grd.set_actnum(actnum)
    newactive = grd.ncol * grd.nrow * grd.nlay - 2 * (grd.ncol * grd.nrow)
    assert grd.nactive == newactive, "Changed ACTNUM"
    grd.to_file(tmp_path / "reek_new_actnum.roff")


def test_eclgrid_import3(tmp_path):
    """Eclipse GRDECL import and translate."""
    grd = xtgeo.grid_from_file(BRILGRDECL, fformat="grdecl")

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


def test_geometrics_reek():
    """Import Reek and test geometrics."""
    grd = xtgeo.grid_from_file(REEKFILE, fformat="egrid")

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


def test_npvalues1d():
    """Different ways of getting np arrays."""
    grd = xtgeo.grid_from_file(DUALFIL3)
    dz = grd.get_dz()

    dz1 = dz.get_npvalues1d(activeonly=False)  # [  1.   1.   1.   1.   1.  nan  ...]
    dz2 = dz.get_npvalues1d(activeonly=True)  # [  1.   1.   1.   1.   1.  1. ...]

    assert dz1[0] == 1.0
    assert np.isnan(dz1[5])
    assert dz1[0] == 1.0
    assert not np.isnan(dz2[5])

    grd = xtgeo.grid_from_file(DUALFIL1)  # all cells active
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


def test_xyz_cell_corners():
    """Test xyz variations."""
    grd = xtgeo.grid_from_file(DUALFIL1)

    allcorners = grd.get_xyz_corners()
    assert len(allcorners) == 24
    assert allcorners[0].get_npvalues1d()[0] == 0.0
    assert allcorners[23].get_npvalues1d()[-1] == 1001.0


def test_grid_layer_slice():
    """Test grid slice coordinates."""
    grd = xtgeo.grid_from_file(REEKFILE)

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


def test_gridquality_properties(show_plot):
    """Get grid quality props."""
    grd1 = xtgeo.grid_from_file(GRIDQC1)

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

    grd2 = xtgeo.grid_from_file(GRIDQC2)
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


def test_bulkvol():
    """Test cell bulk volume calculation."""
    grd = xtgeo.grid_from_file(GRIDQC1)
    cellvol_rms = xtgeo.gridproperty_from_file(GRIDQC1_CELLVOL)

    bulk = grd.get_bulk_volume()
    logger.info("Sum this: %s", bulk.values.sum())
    logger.info("Sum RMS: %s", cellvol_rms.values.sum())


@pytest.mark.benchmark(group="bulkvol")
def test_benchmark_bulkvol(benchmark):
    dimens = (10, 50, 5)
    grd = xtgeo.create_box_grid(dimension=dimens)

    def run():
        _ = grd.get_bulk_volume()

    benchmark(run)


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


@given(xtgeo_grids)
def test_get_dxdy_is_get_dx_and_dy(grid):
    assert np.all(grid.get_dxdy(asmasked=True)[0].values == grid.get_dx().values)
    assert np.all(grid.get_dxdy(asmasked=True)[1].values == grid.get_dy().values)


def test_benchmark_grid_get_dz(benchmark):
    grd = xtgeo.create_box_grid(dimension=(100, 100, 100))

    def run():
        grd.get_dz()

    benchmark(run)


def test_benchmark_grid_get_dxdy(benchmark):
    grd = xtgeo.create_box_grid(dimension=(100, 100, 100))

    def run():
        grd.get_dxdy()

    benchmark(run)


def test_grid_get_dxdydz_zero_size():
    grd = xtgeo.create_box_grid(dimension=(0, 0, 0))

    assert grd.get_dx().values.shape == (0, 0, 0)
    assert grd.get_dy().values.shape == (0, 0, 0)
    assert grd.get_dz().values.shape == (0, 0, 0)


def test_grid_get_dxdydz_bad_coordsv_size():
    grd = xtgeo.create_box_grid(dimension=(10, 10, 10))
    grd._coordsv = np.zeros(shape=(0, 0, 0))

    with pytest.raises(xtgeo.XTGeoCLibError, match="Incorrect size of coordsv"):
        grd.get_dx()
    with pytest.raises(xtgeo.XTGeoCLibError, match="Incorrect size of coordsv"):
        grd.get_dy()
    with pytest.raises(xtgeo.XTGeoCLibError, match="Incorrect size of coordsv"):
        grd.get_dz()


def test_grid_get_dxdydz_bad_zcorn_size():
    grd = xtgeo.create_box_grid(dimension=(10, 10, 10))
    grd._zcornsv = np.zeros(shape=(0, 0, 0, 0))

    with pytest.raises(xtgeo.XTGeoCLibError, match="Incorrect size of zcornsv"):
        grd.get_dx()
    with pytest.raises(xtgeo.XTGeoCLibError, match="Incorrect size of zcornsv"):
        grd.get_dy()
    with pytest.raises(xtgeo.XTGeoCLibError, match="Incorrect size of zcornsv"):
        grd.get_dz()


def test_grid_get_dxdydz_bad_grid_top():
    grd = xtgeo.create_box_grid(dimension=(10, 10, 10))

    grd._coordsv[:, :, 2] = 0.0
    grd._coordsv[:, :, 5] = 0.0
    grd._coordsv[:, :, 0] += 1.0

    with pytest.raises(xtgeo.XTGeoCLibError, match="has near zero height"):
        grd.get_dx()
    with pytest.raises(xtgeo.XTGeoCLibError, match="has near zero height"):
        grd.get_dy()
    with pytest.raises(xtgeo.XTGeoCLibError, match="has near zero height"):
        grd.get_dz()


def test_grid_get_dxdydz_bad_metric():
    grd = xtgeo.create_box_grid(dimension=(10, 10, 10))

    with pytest.raises(ValueError, match="Unknown metric"):
        grd.get_dx(metric="foo")
    with pytest.raises(ValueError, match="Unknown metric"):
        grd.get_dy(metric="foo")
    with pytest.raises(ValueError, match="Unknown metric"):
        grd.get_dz(metric="foo")


def test_grid_roff_subgrids_import_regression(tmp_path):
    grid = xtgeo.create_box_grid(dimension=(5, 5, 67))
    grid.subgrids = OrderedDict(
        [
            ("subgrid_0", list(range(1, 21))),
            ("subgrid_1", list(range(21, 53))),
            ("subgrid_2", list(range(53, 68))),
        ]
    )
    grid.to_file(tmp_path / "grid.roff")

    grid2 = xtgeo.grid_from_file(tmp_path / "grid.roff")
    assert grid2.subgrids == OrderedDict(
        [
            ("subgrid_0", range(1, 21)),
            ("subgrid_1", range(21, 53)),
            ("subgrid_2", range(53, 68)),
        ]
    )


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
            import pyvista as pv
        except ImportError:
            warnings.warn("show_plot is active but no pyvista installed")
            return
        grid = pv.ExplicitStructuredGrid(dims, corners)
        grid = grid.compute_connectivity()
        grid.flip_z(inplace=True)

        grid = grid.hide_cells(indi)
        grid.plot(show_edges=True)


def test_get_vtk_geometries_emerald(show_plot):
    grd = xtgeo.grid_from_file(EME1)

    dims, corners, indi = grd.get_vtk_geometries()
    assert corners.mean() == pytest.approx(2132426.94, abs=0.01)

    poro = xtgeo.gridproperty_from_file(EME1PROP, name="PORO")
    porov = poro.get_npvalues1d(order="F")

    if show_plot:
        try:
            import pyvista as pv
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
