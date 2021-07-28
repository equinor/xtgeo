# -*- coding: utf-8 -*-


import glob
from os.path import join

import pytest

import tests.test_common.test_xtg as tsetup
from xtgeo.common import XTGeoDialog
from xtgeo.well import Well, Wells

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

TPATH = xtg.testpathobj

# =========================================================================
# Do tests
# =========================================================================

WFILES = str(TPATH) + "/wells/battle/1/*.rmswell"


@pytest.fixture(name="loadwells1")
def fixture_loadwells1():
    logger.info("Load well 1")
    wlist = []
    for wfile in glob.glob(WFILES):
        wlist.append(Well(wfile))

    return wlist


def test_import_wells(loadwells1):
    """Import wells from file to Wells."""

    mywell_list = loadwells1

    mywells = Wells()
    mywells.wells = mywell_list

    assert "WELL33" in mywells.names


def test_get_dataframe_allwells(loadwells1):
    """Get a single dataframe for all wells"""

    mywell_list = loadwells1

    mywells = Wells()
    mywells.wells = mywell_list

    df = mywells.get_dataframe(filled=True)

    #    assert df.iat[95610, 4] == 345.4128

    logger.debug(df)


@tsetup.plotskipifroxar
def test_quickplot_wells(tmpdir, loadwells1, generate_plot):
    """Import wells from file to Wells and quick plot."""
    if not generate_plot:
        pytest.skip()

    mywell_list = loadwells1

    mywells = Wells()
    mywells.wells = mywell_list
    mywells.quickplot(filename=join(tmpdir, "quickwells.png"))


def test_wellintersections(tmpdir, loadwells1):
    """Find well crossing"""

    mywell_list = loadwells1

    mywells = Wells()
    mywells.wells = mywell_list
    dfr = mywells.wellintersections()
    logger.info(dfr)
    dfr.to_csv(join(tmpdir, "wells_crossings.csv"))


def test_wellintersections_tvdrange_nowfilter(loadwells1):
    """Find well crossing using coarser sampling to Fence"""

    mywell_list = loadwells1

    mywells = Wells()
    mywells.wells = mywell_list
    print("Limit TVD and downsample...")
    mywells.limit_tvd(1300, 1400)
    mywells.downsample(interval=6)
    print("Limit TVD and downsample...DONE")

    dfr = mywells.wellintersections()
    print(dfr)


def test_wellintersections_tvdrange_no_wfilter(loadwells1):
    """Find well crossing using coarser sampling to Fence, no
    wfilter settings.
    """

    mywell_list = loadwells1

    mywells = Wells()
    mywells.wells = mywell_list
    print("Limit TVD and downsample...")
    mywells.limit_tvd(1300, 1400)
    mywells.downsample(interval=6)
    print("Limit TVD and downsample...DONE")

    dfr = mywells.wellintersections()
    print(dfr)


def test_wellintersections_tvdrange_wfilter(tmpdir, loadwells1):
    """Find well crossing using coarser sampling to Fence, with
    wfilter settings.
    """

    wfilter = {
        "parallel": {"xtol": 4.0, "ytol": 4.0, "ztol": 2.0, "itol": 10, "atol": 5.0}
    }

    mywell_list = loadwells1

    mywells = Wells()
    mywells.wells = mywell_list
    print("Limit TVD and downsample...")
    mywells.limit_tvd(1300, 1400)
    mywells.downsample(interval=6)
    print("Limit TVD and downsample...DONE")

    dfr = mywells.wellintersections(wfilter=wfilter)
    dfr.to_csv(join(tmpdir, "wells_crossings_filter.csv"))
    print(dfr)
