import unittest
import numpy as np
import os
import os.path
import sys
import logging

from xtgeo.surface import RegularSurface
from xtgeo.common import XTGeoDialog


path = 'TMP'
try:
    os.makedirs(path)
except OSError:
    if not os.path.isdir(path):
        raise

xtg = XTGeoDialog()

try:
    roxenv = int(os.environ['ROXENV'])
except Exception:
    roxenv = 0

print(roxenv)

if roxenv != 1:
    print("Do not run ROXENV tests")
else:
    print("Will run ROXENV tests")

# =============================================================================
# Do tests
# =============================================================================


class TestSurfaceRoxapi(unittest.TestCase):
    """Testing suite for surfaces"""

    def getlogger(self, name):

        # if isinstance(self.logger):
        #     return

        format = xtg.loggingformat

        logging.basicConfig(format=format, stream=sys.stdout)
        logging.getLogger().setLevel(xtg.logginglevel)  # root logger!

        self.logger = logging.getLogger(name)

    def test_getsurface(self):
        """
        get a surface from a RMS project.
        """

        if roxenv == 1:
            self.getlogger('test_getsurface')

            self.logger.info('Simple case...')

            project = "/private/jriv/tmp/fossekall.rms10.0.0"

            x = RegularSurface()
            x.from_roxar(project, name='TopIle', category="DepthSurface")

            x.to_file("TMP/topile.gri")

            self.assertEqual(x.nx, 273, "NX of top Ile from RMS")

            self.assertAlmostEqual(x.values.mean(), 2771.82236, places=3)
        else:
            pass


    # def test_irapasc_export(self):
    #     """
    #     Export Irap ASCII (1).
    #     """

    #     self.getlogger('test_irapasc_export')
    #     self.logger.info('Export to Irap Classic')
    #     x = RegularSurface()
    #     x.to_file('TMP/irap.fgr', fformat="irap_ascii")

    #     fstatus = False
    #     if os.path.isfile('TMP/irap.fgr') is True:
    #         fstatus = True

    #     self.assertTrue(fstatus)

    # def test_irapasc_exp2(self):
    #     """
    #     Export Irap ASCII (2).
    #     """
    #     self.getlogger('test_irapasc_exp2')
    #     self.logger.info('Export to Irap Classic and Binary')

    #     x = RegularSurface(
    #         nx=120,
    #         ny=100,
    #         xori=1000,
    #         yori=5000,
    #         xinc=40,
    #         yinc=20,
    #         values=np.random.rand(
    #             120,
    #             100))
    #     self.assertEqual(x.nx, 120)
    #     x.to_file('TMP/irap2_a.fgr', fformat="irap_ascii")
    #     x.to_file('TMP/irap2_b.gri', fformat="irap_binary")

    #     fsize = os.path.getsize('TMP/irap2_b.gri')
    #     self.logger.info(fsize)
    #     self.assertEqual(fsize, 48900)

    # def test_irapbin_io(self):
    #     """
    #     Import and export Irap binary
    #     """
    #     self.getlogger('test_irapbin_io')
    #     self.logger.info('Import and export...')

    #     x = RegularSurface()
    #     x.from_file('../../testdata/Surface/Etc/fossekall1.irapbin',
    #                 fformat='irap_binary')

    #     self.logger.debug("NX is {}".format(x.nx))

    #     self.assertEqual(x.nx, 58)
    #     zval = x.values
    #     self.logger.info('VALUES are:\n{}'.format(zval))

    #     self.logger.info('MEAN value (original):\n{}'.format(zval.mean()))

    #     # add value via numpy
    #     zval = zval + 300
    #     # update
    #     x.values = zval

    #     self.logger.info('MEAN value (update):\n{}'.format(x.values.mean()))

    #     self.assertAlmostEqual(x.values.mean(), 2882.741, places=2)

    #     x.to_file('TMP/foss1_plus_300_a.fgr', fformat='irap_ascii')
    #     x.to_file('TMP/foss1_plus_300_b.gri', fformat='irap_binary')

    #     mfile = '../../testdata/Surface/Etc/fossekall1.irapbin'

    #     # direct import
    #     y = RegularSurface(mfile)
    #     self.assertEqual(y.nx, 58)

    #     # semidirect import
    #     cc = RegularSurface().from_file(mfile)
    #     self.assertEqual(cc.nx, 58)

    # def test_similarity(self):
    #     """
    #     Testing similarity of two surfaces. 0.0 means identical in
    #     terms of mean value.
    #     """

    #     self.getlogger('test_similarity')
    #     self.logger.info('Test if surfaces are similar...')

    #     mfile = '../../testdata/Surface/Etc/fossekall1.irapbin'

    #     x = RegularSurface(mfile)
    #     y = RegularSurface(mfile)

    #     si = x.similarity_index(y)
    #     self.assertEqual(si, 0.0)

    #     y.values = y.values * 2

    #     si = x.similarity_index(y)
    #     self.assertEqual(si, 1.0)

    # def test_irapbin_io_loop(self):
    #     """
    #     Do a loop over big Troll data set
    #     """

    #     self.getlogger('test_irapbin_io_loop')
    #     n = 10
    #     self.logger.info("Import and export map to numpy {} times".format(n))

    #     for i in range(0, 10):
    #         # print(i)
    #         x = RegularSurface()
    #         x.from_file('../../testdata/Surface/T/troll.irapbin',
    #                     fformat='irap_binary')

    #         self.logger.info('Map dimensions: {} {}'.format(x.nx, x.ny))

    #         m1 = x.values.mean()
    #         zval = x.values
    #         zval = zval + 300
    #         x.values = zval
    #         m2 = x.values.mean()
    #         x.to_file('TMP/troll.gri', fformat='irap_binary')
    #         self.logger.info("Mean before and after: {} .. {}".format(m1, m2))

    # #     xtg.info("Import and export map to numpy {} times DONE".format(n))

    # def test_distance_from_point(self):
    #     """
    #     Distance from point
    #     """
    #     self.getlogger('test_distance_from_point')

    #     x = RegularSurface()
    #     x.from_file('../../testdata/Surface/Etc/fossekall1.irapbin',
    #                 fformat='irap_binary')

    #     x.distance_from_point(point=(464960, 7336900), azimuth=30)

    #     x.to_file('TMP/foss1_dist_point.gri', fformat='irap_binary')

    # def test_value_from_xy(self):
    #     """
    #     get Z value from XY point
    #     """
    #     self.getlogger('test_value_from_xy')

    #     x = RegularSurface()
    #     x.from_file('../../testdata/Surface/Etc/fossekall1.irapbin',
    #                 fformat='irap_binary')

    #     z = x.get_value_from_xy(point=(464375.992, 7337128.076))

    #     z = float("{:6.2f}".format(z))

    #     self.assertEqual(z, 2766.96)

    #     # outside value shall return None
    #     z = x.get_value_from_xy(point=(0.0, 7337128.076))
    #     self.assertIsNone(z)


if __name__ == '__main__':

    logging.basicConfig(stream=sys.stderr)
    logging.getLogger('').setLevel(logging.DEBUG)

    print()
    unittest.main()
