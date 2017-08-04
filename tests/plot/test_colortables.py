import unittest
import sys
import logging
import xtgeo.plot._colortables as ct
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

# =========================================================================
# Do tests
# =========================================================================


class TestColors(unittest.TestCase):
    """Testing suite for colo(u)rs"""

    def getlogger(self, name):

        format = xtg.loggingformat

        logging.basicConfig(format=format, stream=sys.stdout)
        logging.getLogger().setLevel(xtg.logginglevel)  # root logger!

        self.logger = logging.getLogger(name)

    def test_readfromfile(self):
        """Read color table from RMS file."""

        cfile = '../../testdata/Various/colfacies2.txt'
        self.getlogger('test_readfromfile')

        ctable = ct.colorsfromfile(cfile)

        self.assertEqual(ctable[5], (0.49019608, 0.38431373, 0.05882353))

        self.logger.info(ctable)

    def test_xtgeo_colors(self):
        """Read the XTGeo color table."""

        self.getlogger('test_xtgeo_colors')

        ctable = ct.xtgeocolors()

        self.assertEqual(ctable[5], (0.000, 1.000, 1.000))

        self.logger.info(ctable)



if __name__ == '__main__':

    unittest.main()
