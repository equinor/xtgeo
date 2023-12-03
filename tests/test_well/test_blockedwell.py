from os.path import join

import pytest
import xtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

testpath = xtg.testpathobj

wfile = join(testpath, "wells/reek/1/OP_1.bw")


@pytest.fixture(name="loadwell1")
def fixture_loadwell1():
    """Fixture for loading a well (pytest setup)"""
    logger.info("Load well 1")
    return xtgeo.blockedwell_from_file(wfile)


def test_import_blockedwell(loadwell1):
    """Import blocked well from file."""

    mywell = loadwell1

    assert mywell.xpos == 461809.6, "XPOS"
    assert mywell.ypos == 5932990.4, "YPOS"
    assert mywell.wellname == "OP_1", "WNAME"
    assert mywell.xname == "X_UTME"

    assert mywell.get_logtype("Facies") == "DISC"
    assert mywell.get_logrecord("Facies") == {
        0: "Background",
        1: "Channel",
        2: "Crevasse",
    }

    assert mywell.dataframe["Poro"][4] == pytest.approx(0.224485, abs=0.0001)
