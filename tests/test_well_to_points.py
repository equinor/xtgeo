import unittest
import os
import glob
import sys
import logging
from xtgeo.well import Well
from xtgeo.xyz import Points
from xtgeo.common import XTGeoDialog


path = 'TMP'
try:
    os.makedirs(path)
except OSError:
    if not os.path.isdir(path):
        raise

xtg = XTGeoDialog()

# =========================================================================
# Do tests
# =========================================================================


class Test(unittest.TestCase):
    """Testing suite for wells"""

    def getlogger(self, name):

        # if isinstance(self.logger):
        #     return

        format = xtg.loggingformat

        logging.basicConfig(format=format, stream=sys.stdout)
        logging.getLogger().setLevel(xtg.logginglevel)  # root logger!

        self.logger = logging.getLogger(name)

    def test_wellzone_to_points(self):
        """
        Import well from file and put zone boundaries to a Points object
        """
        self.getlogger('test_wellzone_to_points')

        wfile = "../xtgeo-testdata/wells/tro/1/31_2-E-1_H.w"

        mywell = Well()
        mywell.from_file(wfile)
        self.logger.info("Imported {}".format(wfile))

        # get the zpoints which is a Points object
        zpoints = mywell.get_zonation_points(zonelogname="ZONELOG")


if __name__ == '__main__':

    unittest.main()
