# -*- coding: utf-8 -*-
import sys
import xtgeo.plot._colortables as ct
from xtgeo.common import XTGeoDialog
xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    sys.exit(-9)

td = xtg.tmpdir
testpath = xtg.testpath

# =========================================================================
# Do tests
# =========================================================================


def test_readfromfile():
    """Read color table from RMS file."""

    cfile = '../xtgeo-testdata/etc/colortables/colfacies.txt'

    ctable = ct.colorsfromfile(cfile)

    assert ctable[5] == (0.49019608, 0.38431373, 0.05882353)

    logger.info(ctable)


def test_xtgeo_colors():
    """Read the XTGeo color table."""

    ctable = ct.xtgeocolors()

    assert ctable[5] == (0.000, 1.000, 1.000)

    logger.info(ctable)
