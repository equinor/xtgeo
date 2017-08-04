import unittest
import os
import os.path
import sys
import logging

import xtgeo.common.calc as xcalc
from xtgeo.common import XTGeoDialog


path = 'TMP'
try:
    os.makedirs(path)
except OSError:
    if not os.path.isdir(path):
        raise

xtg = XTGeoDialog()

# =============================================================================
# Do tests
# =============================================================================


class TestCommon(unittest.TestCase):
    """Testing suite for common routines"""

    def getlogger(self, name):

        # if isinstance(self.logger):
        #     return

        format = xtg.loggingformat

        logging.basicConfig(format=format, stream=sys.stdout)
        logging.getLogger().setLevel(xtg.logginglevel)  # root logger!

        self.logger = logging.getLogger(name)

    def test_ijk_to_ib(self):
        """
        Convert I J K to IB index
        """

        self.getlogger('test_ijk_to_ib')

        ib = xcalc.ijk_to_ib(2, 2, 2, 3, 4, 5)
        self.logger.info(ib)
        self.assertEqual(ib, 16)

    def test_ib_to_ijk(self):
        """
        Convert IB index to IJK tuple.
        """

        self.getlogger('test_ib_to_ijk')

        ijk = xcalc.ib_to_ijk(16, 3, 4, 5)
        self.logger.info(ijk)
        self.assertEqual(ijk[0], 2)


if __name__ == '__main__':
    unittest.main()
