#!/usr/bin/env python -u

import unittest
import os
import sys
import logging

from xtgeo.grid3d import Grid
from xtgeo.well import Well
from xtgeo.grid3d import GridProperty
from xtgeo.common import XTGeoDialog

path = 'TMP'
try:
    os.makedirs(path)
except OSError:
    if not os.path.isdir(path):
        raise

# set default level
xtg = XTGeoDialog()

# =============================================================================
# Do tests
# =============================================================================


class TestGridWellProperty(unittest.TestCase):
    """Testing suite for 3D grid vs Well operations"""

    def getlogger(self, name):

        # if isinstance(self.logger):
        #     return

        format = xtg.loggingformat

        logging.basicConfig(format=format, stream=sys.stdout)
        logging.getLogger().setLevel(xtg.logginglevel)  # root logger!

        self.logger = logging.getLogger(name)

    def test_report_zlog_mismatch(self):
        """
        Report zone log mismatch grid and well
        """
        self.getlogger(sys._getframe(1).f_code.co_name)

        self.logger.info('Name is {}'.format(__name__))
        g1 = Grid()
        g1.from_file('../xtgeo-testdata/3dgrids/gfb/gullfaks2.roff')

        g2 = Grid()
        g2.from_file('../xtgeo-testdata/3dgrids/gfb/gullfaks2.roff')

        g2.reduce_to_one_layer()

        z = GridProperty()
        z.from_file('../xtgeo-testdata/3dgrids/gfb/gullfaks2_zone.roff',
                    name='Zone')

        # w1 = Well()
        # w1.from_file('../xtgeo-testdata/wells/gfb/1/34_10-A-42.w')

        w2 = Well()
        w2.from_file('../xtgeo-testdata/wells/gfb/1/34_10-1.w')

        w3 = Well()
        w3.from_file('../xtgeo-testdata/wells/gfb/1/34_10-B-21_B.w')

        wells = [w2, w3]

        for w in wells:
            response = g1.report_zone_mismatch(
                well=w, zonelogname='ZONELOG', mode=0, zoneprop=z,
                onelayergrid=g2, zonelogrange=[0, 19], option=0,
                depthrange=[1700, 9999])

            if response is None:
                continue
            else:
                self.logger.info(response)
            # if w.wellname == w1.wellname:
            #     match = float("{0:.2f}".format(response[0]))
            #     self.logger.info(match)
            #     self.assertEqual(match, 72.39, 'Match ration A42')

        # perforation log instead

        for w in wells:

            response = g1.report_zone_mismatch(
                well=w, zonelogname='ZONELOG', mode=0, zoneprop=z,
                onelayergrid=g2, zonelogrange=[0, 19], option=0,
                perflogname='PERFLOG')

            if response is None:
                continue
            else:
                self.logger.info(response)

            # if w.wellname == w1.wellname:
            #     match = float("{0:.1f}".format(response[0]))
            #     self.logger.info(match)
            #     self.assertEqual(match, 87.70, 'Match ratio A42')

            if w.wellname == w3.wellname:
                match = float("{0:.2f}".format(response[0]))
                self.logger.info(match)
                self.assertEqual(match, 0.0, 'Match ratio B21B')


if __name__ == '__main__':

    unittest.main()
