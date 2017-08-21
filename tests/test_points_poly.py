import unittest
import os
import glob
import sys
import logging
from xtgeo.xyz import Points
from xtgeo.xyz import Polygons
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

    def test_import(self):
        """
        Import XYZ points from file
        """
        self.getlogger('test_import')

        pfile = "../xtgeo-testdata/points/eme/1/emerald_10_random.poi"

        mypoints = Points()

        mypoints.from_file(pfile)

        self.logger.debug(mypoints.dataframe)

    def test_import_polygons(self):
        """
        Import XYZ polygons from file
        """
        self.getlogger('test_import_polygons')

        pfile = "../xtgeo-testdata/points/eme/1/emerald_10_random.poi"

        mypoly = Polygons()

        mypoly.from_file(pfile)

        self.logger.debug(mypoly.dataframe)

if __name__ == '__main__':

    unittest.main()
