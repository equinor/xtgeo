# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function

import os
from os.path import join
from collections import OrderedDict


import math


import pytest
import numpy as np

import xtgeo
from xtgeo import pathlib
from xtgeo.grid3d import Grid
from xtgeo.grid3d import GridProperty
from xtgeo.common import XTGeoDialog
import test_common.test_xtg as tsetup

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

TMPDIR = xtg.tmpdir
TESTPATH = xtg.testpath

EMEGFILE = "../xtgeo-testdata/3dgrids/eme/1/emerald_hetero_grid.roff"
REEKFILE = "../xtgeo-testdata/3dgrids/reek/REEK.EGRID"
REEKFIL2 = "../xtgeo-testdata/3dgrids/reek3/reek_sim.grdecl"  # ASCII GRDECL
REEKFIL3 = "../xtgeo-testdata/3dgrids/reek3/reek_sim.bgrdecl"  # binary GRDECL
REEKFIL4 = "../xtgeo-testdata/3dgrids/reek/reek_geo_grid.roff"
REEKFIL5 = "../xtgeo-testdata/3dgrids/reek/reek_geo2_grid_3props.roff"
REEKROOT = "../xtgeo-testdata/3dgrids/reek/REEK"
# brilfile = '../xtgeo-testdata/3dgrids/bri/B.GRID' ...disabled
BRILGRDECL = "../xtgeo-testdata/3dgrids/bri/b.grdecl"
BANAL6 = "../xtgeo-testdata/3dgrids/etc/banal6.roff"

DUALFIL1 = "../xtgeo-testdata/3dgrids/etc/dual_grid.roff"
DUALFIL2 = "../xtgeo-testdata/3dgrids/etc/dual_grid_noactivetag.roff"

DUALFIL3 = "../xtgeo-testdata/3dgrids/etc/TEST_DPDK.EGRID"

# =============================================================================
# Do tests
# =============================================================================
# pylint: disable=redefined-outer-name


@pytest.fixture()
def load_gfile1():
    """Fixture for loading EMEGFILE grid"""
    return xtgeo.grid3d.Grid(EMEGFILE)


def test_import_wrong():
    """Importing wrong fformat, etc"""
    with pytest.raises(ValueError):
        grd = Grid()
        grd.from_file(EMEGFILE, fformat="stupid_wrong_name")
        tsetup.assert_equal(grd.ncol, 70)


def test_import_guess(load_gfile1):
    """Import with guessing fformat, and also test name attribute"""

    grd = load_gfile1

    tsetup.assert_equal(grd.ncol, 70)
    tsetup.assert_equal(grd.name, "emerald_hetero_grid")

    grd.name = "xxx"
    tsetup.assert_equal(grd.name, "xxx")


def test_create_shoebox():
    """Make a shoebox grid from scratch"""

    grd = xtgeo.Grid()
    grd.create_box()
    grd.to_file(join(TMPDIR, "shoebox_default.roff"))

    grd.create_box(flip=-1)
    grd.to_file(join(TMPDIR, "shoebox_default_flipped.roff"))

    timer1 = xtg.timer()
    grd.create_box(
        origin=(0, 0, 1000), dimension=(300, 200, 30), increment=(20, 20, 1), flip=-1
    )
    logger.info("Making a a 1,8 mill cell grid took %5.3f secs", xtg.timer(timer1))

    dx, dy = grd.get_dxdy()

    tsetup.assert_almostequal(dx.values.mean(), 20.0, 0.0001)
    tsetup.assert_almostequal(dy.values.mean(), 20.0, 0.0001)

    grd.create_box(
        origin=(0, 0, 1000), dimension=(30, 30, 3), rotation=45, increment=(20, 20, 1)
    )

    x, y, z = grd.get_xyz()

    tsetup.assert_almostequal(x.values1d[0], 0.0, 0.001)
    tsetup.assert_almostequal(y.values1d[0], 20 * math.cos(45 * math.pi / 180), 0.001)
    tsetup.assert_almostequal(z.values1d[0], 1000.5, 0.001)

    grd.create_box(
        origin=(0, 0, 1000),
        dimension=(30, 30, 3),
        rotation=45,
        increment=(20, 20, 1),
        oricenter=True,
    )

    x, y, z = grd.get_xyz()

    tsetup.assert_almostequal(x.values1d[0], 0.0, 0.001)
    tsetup.assert_almostequal(y.values1d[0], 0.0, 0.001)
    tsetup.assert_almostequal(z.values1d[0], 1000.0, 0.001)


def test_shoebox_egrid():
    """Test the egrid format for different grid sizes"""

    dimens = [(1000, 1, 1), (1000, 1, 200), (300, 200, 30)]

    for dim in dimens:
        grd = xtgeo.Grid()
        grd.create_box(dimension=dim)
        grd.to_file(join(TMPDIR, "E1.EGRID"), fformat="egrid")
        grd1 = xtgeo.Grid(join(TMPDIR, "E1.EGRID"))
        assert grd1.dimensions == dim


def test_roffbin_get_dataframe_for_grid(load_gfile1):
    """Import ROFF grid and return a grid dataframe (no props)"""

    grd = load_gfile1

    assert isinstance(grd, Grid)

    df = grd.dataframe()
    print(df.head())

    assert len(df) == grd.nactive

    tsetup.assert_almostequal(df["X_UTME"][0], 459176.7937727844, 0.1)

    assert len(df.columns) == 6

    df = grd.dataframe(activeonly=False)
    print(df.head())

    assert len(df.columns) == 7
    assert len(df) != grd.nactive

    assert len(df) == grd.ncol * grd.nrow * grd.nlay


def test_subgrids(load_gfile1):
    """Import ROFF and test different subgrid functions."""

    grd = load_gfile1

    assert isinstance(grd, Grid)

    logger.info(grd.subgrids)

    newsub = OrderedDict()
    newsub["XX1"] = 20
    newsub["XX2"] = 2
    newsub["XX3"] = 24

    grd.set_subgrids(newsub)
    logger.info(grd.subgrids)

    subs = grd.get_subgrids()
    logger.info(subs)

    assert subs == newsub

    _i_index, _j_index, k_index = grd.get_ijk()

    zprop = k_index.copy()
    zprop.values[k_index.values > 4] = 2
    zprop.values[k_index.values <= 4] = 1
    print(zprop.values)
    grd.describe()
    grd.subgrids_from_zoneprop(zprop)

    grd.describe()


def test_roffbin_import1(load_gfile1):
    """Test roff binary import case 1"""

    grd = load_gfile1

    tsetup.assert_equal(grd.ncol, 70, txt="Grid NCOL Emerald")
    tsetup.assert_equal(grd.nlay, 46, txt="Grid NLAY Emerald")

    # extract ACTNUM parameter as a property instance (a GridProperty)
    act = grd.get_actnum()

    # get dZ...
    dzv = grd.get_dz()

    logger.info("ACTNUM is %s", act)
    logger.debug("DZ values are \n%s", dzv.values1d[888:999])

    dzval = dzv.values
    print("DZ mean and shape: ", dzval.mean(), dzval.shape)
    # # get the value is cell 32 73 1 shall be 2.761
    # mydz = float(dzval[31:32, 72:73, 0:1])
    # tsetup.assert_almostequal(mydz, 2.761, 0.001, txt='Grid DZ Emerald')

    # # get dX dY
    # logger.info('Get dX dY')
    # dx, dy = g.get_dxdy()

    # mydx = float(dx.values3d[31:32, 72:73, 0:1])
    # mydy = float(dy.values3d[31:32, 72:73, 0:1])

    # tsetup.assert_almostequal(mydx, 118.51, 0.01, txt='Grid DX Emerald')
    # tsetup.assert_almostequal(mydy, 141.21, 0.01, txt='Grid DY Emerald')

    # # get X Y Z coordinates (as GridProperty objects) in one go
    # logger.info('Get X Y Z...')
    # x, y, z = g.get_xyz(names=['xxx', 'yyy', 'zzz'])

    # logger.info('X is {}'.format(act))
    # logger.debug('X values are \n{}'.format(x.values[888:999]))

    # tsetup.assert_equal(x.name, 'xxx', txt='Name of X coord')
    # x.name = 'Xerxes'

    # logger.info('X name is now {}'.format(x.name))

    # logger.info('Y is {}'.format(act))
    # logger.debug('Y values are \n{}'.format(y.values[888:999]))

    # # attach some properties to grid
    # g.props = [x, y]

    # logger.info(g.props)
    # g.props = [z]

    # logger.info(g.props)

    # g.props.append(x)
    # logger.info(g.propnames)

    # # get the property of name Xerxes
    # myx = g.get_prop_by_name('Xerxes')
    # if myx is None:
    #     logger.info(myx)
    # else:
    #     logger.info("Got nothing!")


# def test_eclgrid_import1():
#     """Eclipse GRID import."""

#     g = Grid()
#     logger.info("Import Eclipse GRID...")
#     g.from_file(brilfile, fformat="grid")

#     tsetup.assert_equal(g.ncol, 20, txt='Grid NCOL from Eclipse')
#     tsetup.assert_equal(g.nrow, 15, txt='Grid NROW from Eclipse')


# def test_eclgrid_import1_cells():
#     """Eclipse GRID import, test for cell corners."""

#     g = Grid()
#     logger.info("Import Eclipse GRID...")
#     g.from_file(brilfile, fformat="grid")

#     corners = g.get_xyz_cell_corners(ijk=(6, 8, 1))

#     tsetup.assert_almostequal(corners[0], 5071.91, 0.1)
#     tsetup.assert_almostequal(corners[1], 7184.34, 0.1)
#     tsetup.assert_almostequal(corners[2], 7274.81, 0.1)

#     tsetup.assert_almostequal(corners[21], 5995.31, 0.1)
#     tsetup.assert_almostequal(corners[22], 7893.03, 0.1)
#     tsetup.assert_almostequal(corners[23], 7228.98, 0.1)

#     allcorners = g.get_xyz_corners()
#     for corn in allcorners:
#         logger.info(corn.name)

#     logger.info(allcorners[0].values[5, 7, 0])  # x for corn0 at 6, 8, 1
#     logger.info(allcorners[1].values[5, 7, 0])  # y for corn0 at 6, 8, 1
#     logger.info(allcorners[2].values[5, 7, 0])  # z for corn0 at 6, 8, 1

#     tsetup.assert_equal(corners[1], allcorners[1].values[5, 7, 0])
#     tsetup.assert_equal(corners[22], allcorners[22].values[5, 7, 0])


def test_roffbin_import_v2_emerald():
    """Test roff binary import ROFF using new API, emerald"""

    t0 = xtg.timer()
    grd1 = Grid(EMEGFILE)
    tsetup.assert_equal(grd1.ncol, 70)
    print("V2: ", xtg.timer(t0))


@tsetup.bigtest
def test_roffbin_import_v2stress():
    """Test roff binary import ROFF using new API, comapre timing etc"""

    t0 = xtg.timer()
    for _ino in range(100):
        grd1 = Grid()
        grd1.from_file(REEKFIL4)
    t1 = xtg.timer(t0)
    print("100 loops with ROXAPIV 2 took: ", t1)


def test_roffbin_banal6():
    """Test roff binary for banal no. 6 case"""

    grd1 = Grid()
    grd1.from_file(BANAL6)

    grd2 = Grid()
    grd2.from_file(BANAL6, _xtgformat=2)

    assert grd1.get_xyz_cell_corners() == grd2.get_xyz_cell_corners()

    assert grd1.get_xyz_cell_corners((4, 2, 3)) == grd2.get_xyz_cell_corners((4, 2, 3))

    grd2._convert_xtgformat2to1()

    assert grd1.get_xyz_cell_corners((4, 2, 3)) == grd2.get_xyz_cell_corners((4, 2, 3))


@tsetup.bigtest
def test_roffbin_bigbox(tmpdir):
    """Test roff binary for bigbox, to monitor performance"""

    bigbox = os.path.join(tmpdir, "bigbox.roff")
    if not os.path.exists(bigbox):
        logger.info("Create tmp big roff grid file...")
        grd0 = Grid()
        grd0.create_box(dimension=(500, 500, 500))
        grd0.to_file(bigbox)

    grd1 = Grid()
    t0 = xtg.timer()
    grd1.from_file(bigbox, _xtgformat=1)
    t_old = xtg.timer(t0)
    logger.info("Reading bigbox xtgeformat=1 took %s seconds", t_old)
    cell1 = grd1.get_xyz_cell_corners((13, 14, 15))

    grd2 = Grid()
    t0 = xtg.timer()
    t1 = xtg.timer()
    grd2.from_file(bigbox, _xtgformat=2)
    t_new = xtg.timer(t0)
    logger.info("Reading bigbox xtgformat=2 took %s seconds", t_new)
    cell2a = grd2.get_xyz_cell_corners((13, 14, 15))

    t0 = xtg.timer()
    grd2._convert_xtgformat2to1()
    cell2b = grd2.get_xyz_cell_corners((13, 14, 15))
    logger.info("Conversion to xtgformat1 took %s seconds", xtg.timer(t0))
    t_newtot = xtg.timer(t1)
    logger.info("Total run time xtgformat=2 + conv took %s seconds", t_newtot)

    logger.info("Speed gain new vs old: %s", t_old / t_new)
    logger.info("Speed gain new incl conv vs old: %s", t_old / t_newtot)

    assert cell1 == cell2a
    assert cell1 == cell2b


def test_roffbin_import_v2_wsubgrids():
    """Test roff binary import ROFF using new API, now with subgrids"""

    grd1 = Grid()
    grd1.from_file(REEKFIL5)
    print(grd1.subgrids)


def test_import_grdecl_and_bgrdecl():
    """Eclipse import of GRDECL and binary GRDECL"""
    grd1 = Grid(REEKFIL2, fformat="grdecl")

    grd1.describe()
    assert grd1.dimensions == (40, 64, 14)
    assert grd1.nactive == 35812

    # get dZ...
    dzv1 = grd1.get_dz()

    grd2 = Grid(REEKFIL3, fformat="bgrdecl")

    grd2.describe()
    assert grd2.dimensions == (40, 64, 14)
    assert grd2.nactive == 35812

    # get dZ...
    dzv2 = grd2.get_dz()

    tsetup.assert_almostequal(dzv1.values.mean(), dzv2.values.mean(), 0.001)


def test_eclgrid_import2():
    """Eclipse EGRID import, also change ACTNUM."""
    grd = Grid()
    logger.info("Import Eclipse GRID...")
    grd.from_file(REEKFILE, fformat="egrid")

    tsetup.assert_equal(grd.ncol, 40, txt="EGrid NX from Eclipse")
    tsetup.assert_equal(grd.nrow, 64, txt="EGrid NY from Eclipse")
    tsetup.assert_equal(grd.nactive, 35838, txt="EGrid NTOTAL from Eclipse")
    tsetup.assert_equal(grd.ntotal, 35840, txt="EGrid NACTIVE from Eclipse")

    actnum = grd.get_actnum()
    print(actnum.values[12:13, 22:24, 5:6])
    tsetup.assert_equal(actnum.values[12, 22, 5], 0, txt="ACTNUM 0")

    actnum.values[:, :, :] = 1
    actnum.values[:, :, 4:6] = 0
    grd.set_actnum(actnum)
    newactive = grd.ncol * grd.nrow * grd.nlay - 2 * (grd.ncol * grd.nrow)
    tsetup.assert_equal(grd.nactive, newactive, txt="Changed ACTNUM")
    grd.to_file(join(TMPDIR, "reek_new_actnum.roff"))


def test_eclgrid_import3():
    """Eclipse GRDECL import and translate"""

    grd = Grid(BRILGRDECL, fformat="grdecl")

    mylist = grd.get_geometrics()

    xori1 = mylist[0]

    # translate the coordinates
    grd.translate_coordinates(translate=(100, 100, 10), flip=(1, 1, 1))

    mylist = grd.get_geometrics()

    xori2 = mylist[0]

    # check if origin is translated 100m in X
    tsetup.assert_equal(xori1 + 100, xori2, txt="Translate X distance")

    grd.to_file(os.path.join(TMPDIR, "g1_translate.roff"), fformat="roff_binary")

    grd.to_file(os.path.join(TMPDIR, "g1_translate.bgrdecl"), fformat="bgrdecl")


def test_geometrics_reek():
    """Import Reek and test geometrics"""

    grd = Grid(REEKFILE, fformat="egrid")

    geom = grd.get_geometrics(return_dict=True, cellcenter=False)

    for key, val in geom.items():
        logger.info("%s is %s", key, val)

    # compared with RMS info:
    tsetup.assert_almostequal(geom["xmin"], 456510.6, 0.1, "Xmin")
    tsetup.assert_almostequal(geom["ymax"], 5938935.5, 0.1, "Ymax")

    # cellcenter True:
    geom = grd.get_geometrics(return_dict=True, cellcenter=True)
    tsetup.assert_almostequal(geom["xmin"], 456620, 1, "Xmin cell center")


def test_activate_all_cells():
    """Make the grid active for all cells"""

    grid = Grid(EMEGFILE)
    logger.info("Number of active cells %s before", grid.nactive)
    grid.activate_all()
    logger.info("Number of active cells %s after", grid.nactive)

    assert grid.nactive == grid.ntotal
    grid.to_file(join(TMPDIR, "emerald_all_active.roff"))


def test_get_adjacent_cells():
    """Get the cell indices for discrete value X vs Y, if connected"""

    grid = Grid(EMEGFILE)
    actnum = grid.get_actnum()
    actnum.to_file(join(TMPDIR, "emerald_actnum.roff"))
    result = grid.get_adjacent_cells(actnum, 0, 1, activeonly=False)
    result.to_file(join(TMPDIR, "emerald_adj_cells.roff"))


def test_simple_io():
    """Test various import and export formats, incl egrid and bgrdecl"""

    gg = Grid(REEKFILE, fformat="egrid")

    assert gg.ncol == 40

    filex = os.path.join(TMPDIR, "grid_test_simple_io.roff")

    gg.to_file(filex)

    gg2 = Grid(filex, fformat="roff")

    assert gg2.ncol == 40

    filex = os.path.join(TMPDIR, "grid_test_simple_io.EGRID")
    filey = os.path.join(TMPDIR, "grid_test_simple_io.bgrdecl")

    gg.to_file(filex, fformat="egrid")
    gg.to_file(filey, fformat="bgrdecl")

    gg2 = Grid(filex, fformat="egrid")
    gg3 = Grid(filey, fformat="bgrdecl")

    assert gg2.ncol == 40

    dz1 = gg.get_dz()
    dz2 = gg2.get_dz()
    dz3 = gg3.get_dz()

    tsetup.assert_almostequal(dz1.values.mean(), dz2.values.mean(), 0.001)
    tsetup.assert_almostequal(dz1.values.std(), dz2.values.std(), 0.001)
    tsetup.assert_almostequal(dz1.values.mean(), dz3.values.mean(), 0.001)
    tsetup.assert_almostequal(dz1.values.std(), dz3.values.std(), 0.001)


def test_ecl_run():
    """Test import an eclrun with dates and export to roff after a diff"""

    dates = [19991201, 20030101]
    rprops = ["PRESSURE", "SWAT"]

    gg = Grid(REEKROOT, fformat="eclipserun", restartdates=dates, restartprops=rprops)

    # get the property object:
    pres1 = gg.get_prop_by_name("PRESSURE_20030101")
    tsetup.assert_almostequal(pres1.values.mean(), 308.45, 0.001)

    pres1.to_file(os.path.join(TMPDIR, "pres1.roff"))

    pres2 = gg.get_prop_by_name("PRESSURE_19991201")

    if isinstance(pres2, GridProperty):
        pass

    logger.debug(pres1.values)
    logger.debug(pres2.values)

    pres1.values = pres1.values - pres2.values
    # logger.debug(pres1.values)
    # logger.debug(pres1)
    avg = pres1.values.mean()
    # ok checked in RMS:
    tsetup.assert_almostequal(avg, -26.073, 0.001)

    pres1.to_file(os.path.join(TMPDIR, "pressurediff.roff"), name="PRESSUREDIFF")


def test_npvalues1d():
    """Different ways of getting np arrays"""

    grd = Grid(DUALFIL3)
    dz = grd.get_dz()

    dz1 = dz.get_npvalues1d(activeonly=False)  # [  1.   1.   1.   1.   1.  nan  ...]
    dz2 = dz.get_npvalues1d(activeonly=True)  # [  1.   1.   1.   1.   1.  1. ...]

    assert dz1[0] == 1.0
    assert np.isnan(dz1[5])
    assert dz1[0] == 1.0
    assert not np.isnan(dz2[5])

    grd = Grid(DUALFIL1)  # all cells active
    dz = grd.get_dz()

    dz1 = dz.get_npvalues1d(activeonly=False)
    dz2 = dz.get_npvalues1d(activeonly=True)

    assert dz1.all() == dz2.all()


def test_pathlib():
    """Import and export via pathlib"""

    pfile = pathlib.Path(DUALFIL1)
    grd = Grid()
    grd.from_file(pfile)

    assert grd.dimensions == (5, 3, 1)

    out = pathlib.Path() / TMPDIR / "grdpathtest.roff"
    grd.to_file(out, fformat="roff")

    with pytest.raises(OSError):
        out = pathlib.Path() / "nosuchdir" / "grdpathtest.roff"
        grd.to_file(out, fformat="roff")


def test_grid_design(load_gfile1):
    """Determine if a subgrid is topconform (T), baseconform (B), proportional (P)

    "design" refers to type of conformity
    "dzsimbox" is avg or representative simbox thickness per cell

    """

    grd = load_gfile1

    print(grd.subgrids)

    code = grd.estimate_design(1)
    assert code["design"] == "P"
    tsetup.assert_almostequal(code["dzsimbox"], 2.5488, 0.001)

    code = grd.estimate_design(2)
    assert code["design"] == "T"
    tsetup.assert_almostequal(code["dzsimbox"], 3.0000, 0.001)

    code = grd.estimate_design("subgrid_0")
    assert code["design"] == "P"

    code = grd.estimate_design("subgrid_1")
    assert code["design"] == "T"

    code = grd.estimate_design("subgrid_2")
    assert code is None

    with pytest.raises(ValueError):
        code = grd.estimate_design(nsub=None)


def test_flip(load_gfile1):
    """Determine if grid is flipped (lefthanded vs righthanded)"""

    grd = load_gfile1

    assert grd.estimate_flip() == 1

    grd.create_box(dimension=(30, 20, 3), flip=-1)
    assert grd.estimate_flip() == -1

    grd.create_box(dimension=(30, 20, 3), rotation=30, flip=-1)
    assert grd.estimate_flip() == -1

    grd.create_box(dimension=(30, 20, 3), rotation=190, flip=-1)
    assert grd.estimate_flip() == -1


def test_xyz_cell_corners():
    """Test xyz variations"""

    grd = Grid(DUALFIL1)

    allcorners = grd.get_xyz_corners()
    assert len(allcorners) == 24
    assert allcorners[0].get_npvalues1d()[0] == 0.0
    assert allcorners[23].get_npvalues1d()[-1] == 1001.0


def test_grid_layer_slice():
    """Test grid slice coordinates"""

    grd = Grid()
    grd.from_file(REEKFILE)

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


def test_generate_hash():
    """Generate hash for two grid instances with same input and compare"""

    grd1 = Grid(REEKFILE)
    grd2 = Grid(REEKFILE)

    assert id(grd1) != id(grd2)

    assert grd1.generate_hash() == grd2.generate_hash()
