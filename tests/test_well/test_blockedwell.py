# -*- coding: utf-8 -*-


from os.path import join

import pytest

from xtgeo.well import BlockedWell
from xtgeo.common import XTGeoDialog

import tests.test_common.test_xtg as tsetup

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

td = xtg.tmpdir
testpath = xtg.testpathobj

# =========================================================================
# Do tests
# =========================================================================

wfile = join(testpath, "wells/reek/1/OP_1.bw")


@pytest.fixture()
def loadwell1():
    """Fixture for loading a well (pytest setup)"""
    logger.info("Load well 1")
    return BlockedWell(wfile)


def test_import(loadwell1):
    """Import well from file."""

    mywell = loadwell1

    print(mywell.dataframe)

    logger.debug("True well name:", mywell.truewellname)
    tsetup.assert_equal(mywell.xpos, 461809.6, "XPOS")
    tsetup.assert_equal(mywell.ypos, 5932990.4, "YPOS")
    tsetup.assert_equal(mywell.wellname, "OP_1", "WNAME")

    logger.info(mywell.get_logtype("Facies"))
    logger.info(mywell.get_logrecord("Facies"))

    # logger.info the numpy string of Poro...
    logger.info(type(mywell.dataframe["Poro"].values))

    dfr = mywell.dataframe
    tsetup.assert_almostequal(dfr["Poro"][4], 0.224485, 0.0001)
