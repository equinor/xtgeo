import unittest
# import numpy as np
import os
import sys
import logging

from xtgeo.grid3d import Grid
from xtgeo.grid3d import GridProperties
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


class TestGridProperties(unittest.TestCase):
    """Testing suite for 3D grid properties (multi)"""

    def getlogger(self, name):

        logging.basicConfig(format=xtg.loggingformat, stream=sys.stdout)
        logging.getLogger().setLevel(xtg.logginglevel)  # root logger!

        self.logger = logging.getLogger(name)

    def test_import_init(self):
        """Import INIT Gullfaks"""

        self.getlogger(sys._getframe(1).f_code.co_name)

        g = Grid()
        g.from_file('../../testdata/Zone/GULLFAKS.EGRID', fformat="egrid")

        x = GridProperties()

        names = ['PORO', 'PORV']
        x.from_file('../../testdata/Zone/GULLFAKS.INIT', fformat="init",
                    names=names, grid=g)

        # get the object
        poro = x.get_prop_by_name('PORO')
        self.logger.info("PORO avg {}".format(poro.values.mean()))

        porv = x.get_prop_by_name('PORV')
        self.logger.info("PORV avg {}".format(porv.values.mean()))
        self.assertAlmostEqual(poro.values.mean(), 0.261157,
                               places=5, msg='Average PORO Gullfaks')

    def test_import_restart(self):
        """Import Restart"""

        self.getlogger(sys._getframe(1).f_code.co_name)

        g = Grid()
        g.from_file('../../testdata/Zone/ECLIPSE.EGRID', fformat="egrid")

        x = GridProperties()

        names = ['PRESSURE', 'SWAT']
        dates = [19851001, 19870701]
        x.from_file('../../testdata/Zone/ECLIPSE.UNRST',
                    fformat="unrst", names=names, dates=dates,
                    grid=g)

        # get the object
        pr = x.get_prop_by_name('PRESSURE_19851001')

        swat = x.get_prop_by_name('SWAT_19851001')

        self.logger.info(x.names)

        self.logger.info(swat.values3d.mean())
        self.logger.info(pr.values3d.mean())

        self.assertAlmostEqual(pr.values.mean(), 332.54578,
                               places=4, msg='Average PRESSURE_19851001')
        self.assertAlmostEqual(swat.values.mean(), 0.87,
                               places=2, msg='Average SWAT_19851001')

        pr = x.get_prop_by_name('PRESSURE_19870701')
        self.logger.info(pr.values3d.mean())
        self.assertAlmostEqual(pr.values.mean(), 331.62,
                               places=2, msg='Average PRESSURE_19870701')

    def test_import_restart_gull(self):
        """Import Restart Gullfaks"""
        self.getlogger(sys._getframe(1).f_code.co_name)

        g = Grid()
        g.from_file('../../testdata/Zone/GULLFAKS.EGRID', fformat="egrid")

        x = GridProperties()

        names = ['PRESSURE', 'SWAT']
        dates = [19851001]
        x.from_file('../../testdata/Zone/GULLFAKS.UNRST',
                    fformat="unrst", names=names, dates=dates,
                    grid=g)

        # get the object
        pr = x.get_prop_by_name('PRESSURE_19851001')

        swat = x.get_prop_by_name('SWAT_19851001')

        self.logger.info(x.names)

        self.logger.info(swat.values3d.mean())
        self.logger.info(pr.values3d.mean())

        # self.assertAlmostEqual(pr.values.mean(), 332.54578,
        #                        places=4, msg='Average PRESSURE_19851001')
        # self.assertAlmostEqual(swat.values.mean(), 0.87,
        #                        places=2, msg='Average SWAT_19851001')

        # pr = x.get_prop_by_name('PRESSURE_19870701')
        # self.logger.info(pr.values3d.mean())
        # self.assertAlmostEqual(pr.values.mean(), 331.62,
        #                        places=2, msg='Average PRESSURE_19870701')

    def test_import_soil(self):
        """SOIL need to be computed in code from SWAT and SGAS"""

        self.getlogger(sys._getframe(1).f_code.co_name)

        g = Grid()
        g.from_file('../../testdata/Zone/ECLIPSE.EGRID', fformat="egrid")

        x = GridProperties()

        names = ['SOIL']
        dates = [19851001]
        x.from_file('../../testdata/Zone/ECLIPSE.UNRST',
                    fformat="unrst", names=names, dates=dates,
                    grid=g)

        # get the object instance
        soil = x.get_prop_by_name('SOIL_19851001')
        self.logger.info(soil.values3d.mean())

        self.logger.debug(x.names)
        self.assertAlmostEqual(soil.values.mean(), 0.1246,
                               places=3, msg='Average SOIL_19850101')


if __name__ == '__main__':

    unittest.main()
