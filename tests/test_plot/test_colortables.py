import pathlib

import xtgeo.plot._colortables as ct
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)


def test_readfromfile(testdata_path):
    """Read color table from RMS file."""

    cfile = testdata_path / pathlib.Path("etc/colortables/colfacies.txt")

    ctable = ct.colorsfromfile(cfile)

    assert ctable[5] == (0.49019608, 0.38431373, 0.05882353)

    logger.info(ctable)


def test_xtgeo_colors():
    """Read the XTGeo color table."""

    ctable = ct.xtgeocolors()

    assert ctable[5] == (0.000, 1.000, 1.000)

    logger.info(ctable)
