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

# =============================================================================
# Do tests
# =============================================================================


class TestSurface(unittest.TestCase):
    """Testing suite for surfaces"""

    def getlogger(self, name):

        # if isinstance(self.logger):
        #     return

        format = xtg.loggingformat

        logging.basicConfig(format=format, stream=sys.stdout)
        logging.getLogger().setLevel(xtg.logginglevel)  # root logger!

        self.logger = logging.getLogger(name)

    def test_create(self):
        """
        Create default surface
        """

        self.getlogger('test_create')

        self.logger.info('Simple case...')

        x = RegularSurface()
        self.assertEqual(x.ncol, 5, 'NX')
        self.assertEqual(x.nrow, 3, 'NY')
        v = x.values
        xdim, ydim = v.shape
        self.assertEqual(xdim, 5, 'NX from DIM')

    def test_irapasc_export(self):
        """
        Export Irap ASCII (1).
        """

        self.getlogger('test_irapasc_export')
        self.logger.info('Export to Irap Classic')
        x = RegularSurface()
        x.to_file('TMP/irap.fgr', fformat="irap_ascii")

        fstatus = False
        if os.path.isfile('TMP/irap.fgr') is True:
            fstatus = True

        self.assertTrue(fstatus)

    def test_irapasc_exp2(self):
        """
        Export Irap ASCII (2).
        """
        self.getlogger('test_irapasc_exp2')
        self.logger.info('Export to Irap Classic and Binary')

        x = RegularSurface(
            ncol=120,
            nrow=100,
            xori=1000,
            yori=5000,
            xinc=40,
            yinc=20,
            values=np.random.rand(
                120,
                100))
        self.assertEqual(x.ncol, 120)
        x.to_file('TMP/irap2_a.fgr', fformat="irap_ascii")
        x.to_file('TMP/irap2_b.gri', fformat="irap_binary")

        fsize = os.path.getsize('TMP/irap2_b.gri')
        self.logger.info(fsize)
        self.assertEqual(fsize, 48900)

    def test_irapbin_io(self):
        """
        Import and export Irap binary
        """
        self.getlogger('test_irapbin_io')
        self.logger.info('Import and export...')

        x = RegularSurface()
        x.from_file('../xtgeo-testdata/surfaces/fos/1/fossekall1.irapbin',
                    fformat='irap_binary')

        self.logger.debug("NX is {}".format(x.ncol))

        self.assertEqual(x.ncol, 58)

        # get the 1D numpy
        v1d = x.get_zval()

        zval = x.values

        self.logger.info('VALUES are:\n{}'.format(zval))

        self.logger.info('MEAN value (original):\n{}'.format(zval.mean()))

        # add value via numpy
        zval = zval + 300
        # update
        x.values = zval

        self.logger.info('MEAN value (update):\n{}'.format(x.values.mean()))

        self.assertAlmostEqual(x.values.mean(), 2882.741, places=2)

        x.to_file('TMP/foss1_plus_300_a.fgr', fformat='irap_ascii')
        x.to_file('TMP/foss1_plus_300_b.gri', fformat='irap_binary')

        mfile = '../xtgeo-testdata/surfaces/fos/1/fossekall1.irapbin'

        # direct import
        y = RegularSurface(mfile)
        self.assertEqual(y.ncol, 58)

        # semidirect import
        cc = RegularSurface().from_file(mfile)
        self.assertEqual(cc.ncol, 58)

    def test_get_xy_value_lists(self):
        """Get the xy list and value list"""
        self.getlogger('test_get_xy_value_lists')

        x = RegularSurface()
        x.from_file('../xtgeo-testdata/surfaces/fos/1/fossekall1.irapbin',
                    fformat='irap_binary')

        xylist, valuelist = x.get_xy_value_lists(valuefmt='8.3f',
                                                 xyfmt='12.2f')

        self.logger.info(xylist[2])
        self.logger.info(valuelist[2])

        self.assertEqual(valuelist[2], 2813.981)

    def test_similarity(self):
        """
        Testing similarity of two surfaces. 0.0 means identical in
        terms of mean value.
        """

        self.getlogger('test_similarity')
        self.logger.info('Test if surfaces are similar...')

        mfile = '../xtgeo-testdata/surfaces/fos/1/fossekall1.irapbin'

        x = RegularSurface(mfile)
        y = RegularSurface(mfile)

        si = x.similarity_index(y)
        self.assertEqual(si, 0.0)

        y.values = y.values * 2

        si = x.similarity_index(y)
        self.assertEqual(si, 1.0)

    def test_irapbin_io_loop(self):
        """
        Do a loop over big Troll data set
        """

        self.getlogger('test_irapbin_io_loop')
        n = 10
        self.logger.info("Import and export map to numpy {} times".format(n))

        for i in range(0, 10):
            # print(i)
            x = RegularSurface()
            x.from_file('../xtgeo-testdata/surfaces/tro/1/troll.irapbin',
                        fformat='irap_binary')

            self.logger.info('Map dimensions: {} {}'.format(x.ncol, x.nrow))

            m1 = x.values.mean()
            zval = x.values
            zval = zval + 300
            x.values = zval
            m2 = x.values.mean()
            x.to_file('TMP/troll.gri', fformat='irap_binary')
            self.logger.info("Mean before and after: {} .. {}".format(m1, m2))

    #     xtg.info("Import and export map to numpy {} times DONE".format(n))

    def test_distance_from_point(self):
        """
        Distance from point
        """
        self.getlogger('test_distance_from_point')

        x = RegularSurface()
        x.from_file('../xtgeo-testdata/surfaces/fos/1/fossekall1.irapbin',
                    fformat='irap_binary')

        x.distance_from_point(point=(464960, 7336900), azimuth=30)

        x.to_file('TMP/foss1_dist_point.gri', fformat='irap_binary')

    def test_value_from_xy(self):
        """
        get Z value from XY point
        """
        self.getlogger('test_value_from_xy')

        x = RegularSurface()
        x.from_file('../xtgeo-testdata/surfaces/fos/1/fossekall1.irapbin',
                    fformat='irap_binary')

        z = x.get_value_from_xy(point=(464375.992, 7337128.076))

        z = float("{:6.2f}".format(z))

        self.assertEqual(z, 2766.96)

        # outside value shall return None
        z = x.get_value_from_xy(point=(0.0, 7337128.076))
        self.assertIsNone(z)

    def test_fence(self):
        """
        Test sampling a fence from a surface.
        """
        self.getlogger('test_fence')

        myfence = np.array(
            [[4.56316489e+05, 6.78292266e+06, 1.00279890e+03, 0.000],
             [4.56299080e+05, 6.78293251e+06, 1.00279890e+03, 20.00],
             [4.56281670e+05, 6.78294235e+06, 1.00279890e+03, 40.00],
             [4.56264261e+05, 6.78295220e+06, 1.00279890e+03, 60.00],
             [4.56246851e+05, 6.78296204e+06, 1.00279890e+03, 80.00],
             [4.56229442e+05, 6.78297188e+06, 1.00279890e+03, 100.0],
             [4.56212032e+05, 6.78298173e+06, 1.00279890e+03, 120.0],
             [4.56194623e+05, 6.78299157e+06, 1.00279890e+03, 140.0],
             [4.56177213e+05, 6.78300142e+06, 1.00279890e+03, 160.0]])


        self.logger.debug("NP:")
        self.logger.debug(myfence)

        x = RegularSurface()
        x.from_file('../xtgeo-testdata/surfaces/fos/1/fossekall1.irapbin',
                    fformat='irap_binary')

        newfence = x.get_fence(myfence)

        self.logger.debug("updated NP:")
        self.logger.debug(newfence)

if __name__ == '__main__':

    logging.basicConfig(stream=sys.stderr)
    logging.getLogger('').setLevel(logging.DEBUG)

    unittest.main()
