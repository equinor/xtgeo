import glob

from os.path import join as ojoin
import pytest

from xtgeo.well import Well
from xtgeo.well import Wells
from xtgeo.common import XTGeoDialog

# import test_common.test_xtg as tsetup

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

td = xtg.tmpdir
testpath = xtg.testpath

# =========================================================================
# Do tests
# =========================================================================

wfiles = "../xtgeo-testdata/wells/battle/1/*.rmswell"


@pytest.fixture()
def loadwells1():
    logger.info('Load well 1')
    wlist = []
    for wfile in glob.glob(wfiles):
        wlist.append(Well(wfile))
    return wlist


def test_import_wells(loadwells1):
    """Import wells from file to Wells."""

    mywell_list = loadwells1

    mywells = Wells()
    mywells.wells = mywell_list

    assert 'WELL33' in mywells.names


def test_quickplot_wells(loadwells1):
    """Import wells from file to Wells and quick plot."""

    mywell_list = loadwells1

    mywells = Wells()
    mywells.wells = mywell_list
    mywells.quickplot(filename=ojoin(td, 'quickwells.png'))


def test_wellintersections(loadwells1):
    """Find well crossing"""

    mywell_list = loadwells1

    mywells = Wells()
    mywells.wells = mywell_list
    dfr = mywells.wellintersections()
    logger.info(dfr)


def test_wellintersections_sampling(loadwells1):
    """Find well crossing using coarser sampling to Fence"""

    mywell_list = loadwells1

    mywells = Wells()
    mywells.wells = mywell_list
    mywells.wellintersections(fencesampling=10.0, tvdrange=(1300, 9999))
    dfr = mywells.wellintersections()
    logger.info(dfr)
