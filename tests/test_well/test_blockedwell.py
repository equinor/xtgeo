# -*- coding: utf-8 -*-


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


@pytest.fixture()
def loadwell1():
    """Fixture for loading a well (pytest setup)"""
    logger.info("Load well 1")
    return xtgeo.blockedwell_from_file(wfile)


def test_import(loadwell1):
    """Import well from file."""

    mywell = loadwell1

    print(mywell.dataframe)

    logger.debug("True well name:", mywell.truewellname)
    assert mywell.xpos == 461809.6, "XPOS"
    assert mywell.ypos == 5932990.4, "YPOS"
    assert mywell.wellname == "OP_1", "WNAME"

    logger.info(mywell.get_logtype("Facies"))
    logger.info(mywell.get_logrecord("Facies"))

    # logger.info the numpy string of Poro...
    logger.info(type(mywell.dataframe["Poro"].values))

    dfr = mywell.dataframe
    assert dfr["Poro"][4] == pytest.approx(0.224485, abs=0.0001)
