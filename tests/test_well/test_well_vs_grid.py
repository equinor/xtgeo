# -*- coding: utf-8 -*-
from __future__ import division, absolute_import
from __future__ import print_function

from os.path import join

import pytest


from xtgeo.well import Well
from xtgeo.grid3d import Grid, GridProperty
from xtgeo.common import XTGeoDialog

import test_common.test_xtg as tsetup

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

TMPDIR = xtg.tmpdir
TESTPATH = xtg.testpath
EQGULLTESTPATH = "../xtgeo-testdata-equinor/data"

# =========================================================================
# Do tests
# pylint: disable=redefined-outer-name
# =========================================================================

WFILE = join(TESTPATH, "wells/reek/1/OP_1.w")
GFILE = join(TESTPATH, "3dgrids/reek/REEK.EGRID")
PFILE = join(TESTPATH, "3dgrids/reek/REEK.INIT")

GGULLFILE = join(EQGULLTESTPATH, "3dgrids/gfb/gullfaks_gg.roff")
WGULLFILE = join(EQGULLTESTPATH, "wells/gfb/1/34_10-A-42.w")


@pytest.fixture()
def loadwell1():
    """Fixture for loading a well (pytest setup)"""
    logger.info("Load well 1")
    return Well(WFILE)


@pytest.fixture()
def loadgrid1():
    """Fixture for loading a grid (pytest setup)"""
    logger.info("Load grid 1")
    return Grid(GFILE)


@pytest.fixture()
def loadporo1(loadgrid1):
    """Fixture for loading a grid poro values (pytest setup)"""
    logger.info("Load PORO 1")
    grd = loadgrid1
    return GridProperty(PFILE, name="PORO", grid=grd)


def test_make_ijk_grid(loadwell1, loadgrid1):
    """Import well from and grid and make I J K logs"""

    mywell = loadwell1
    mygrid = loadgrid1

    mywell.make_ijk_from_grid(mygrid)

    df = mywell.dataframe

    assert int(df.iloc[4850]["ICELL"]) == 29
    assert int(df.iloc[4850]["JCELL"]) == 28
    assert int(df.iloc[4850]["KCELL"]) == 13
    assert int(df.iloc[4847]["KCELL"]) == 12

    assert int(df.iloc[4775]["ICELL"]) == 29
    assert int(df.iloc[4775]["JCELL"]) == 28
    assert int(df.iloc[4775]["KCELL"]) == 1


@tsetup.equinor
@tsetup.bigtest
def test_make_ijk_gf_geogrid():
    """Import well from and a large geogrid and make I J K logs"""

    logger.info("Running test... %s", __name__)
    mywell = Well(WGULLFILE)
    mygrid = Grid(GGULLFILE)

    logger.info("Number of cells in grid is %s", mygrid.ntotal)

    mywell.make_ijk_from_grid(mygrid)

    df = mywell.dataframe

    assert int(df.iloc[16120]["ICELL"]) == 68
    assert int(df.iloc[16120]["JCELL"]) == 204
    assert int(df.iloc[16120]["KCELL"]) == 15


def test_well_get_gridprops(loadwell1, loadgrid1, loadporo1):
    """Import well from and grid and make I J K logs"""

    mywell = loadwell1
    mygrid = loadgrid1
    myporo = loadporo1

    mywell.get_gridproperties(myporo, mygrid)

    myactnum = mygrid.get_actnum()
    myactnum.codes = {0: "INACTIVE", 1: "ACTIVE"}
    myactnum.describe()

    mywell.get_gridproperties(myactnum, mygrid)
    mywell.to_file(join(TMPDIR, "w_from_gprops.w"))
    tsetup.assert_almostequal(mywell.dataframe.iloc[4775]["PORO_model"], 0.2741, 0.001)
    assert mywell.dataframe.iloc[4775]["ACTNUM_model"] == 1
    assert mywell.isdiscrete("ACTNUM_model") is True
