import unittest
import os
import logging
import sys

from xtgeo.cube import Cube
from xtgeo.common import XTGeoDialog


path = 'TMP'
try:
    os.makedirs(path)
except OSError:
    if not os.path.isdir(path):
        raise

xtg = XTGeoDialog()

try:
    bigtest = int(os.environ['BIGTEST'])
except Exception:
    bigtest = 0

# =============================================================================
# Do tests
# =============================================================================


class TestCube(unittest.TestCase):
    """Testing suite for cubes"""

    def getlogger(self, name):

        # if isinstance(self.logger):
        #     return

        format = xtg.loggingformat

        logging.basicConfig(format=format, stream=sys.stdout)
        logging.getLogger().setLevel(xtg.logginglevel)  # root logger!

        self.logger = logging.getLogger(name)

    def test_create(self):
        """Create default cube."""
        x = Cube()
        self.assertEqual(x.nx, 5, 'NX')
        self.assertEqual(x.ny, 3, 'NY')
        v = x.values
        xdim, ydim, zdim = v.shape
        self.assertEqual(xdim, 5, 'NX from DIM')

    def test_segy_scanheader(self):
        """Scan SEGY and report header."""
        self.getlogger(sys._getframe(1).f_code.co_name)
        self.logger.info('Scan header...')

        x = Cube().scan_segy_header('../../testdata/Cube/G/nh0304.segy',
                                    outfile="TMP/cube_scanheader")

        self.logger.info('Scan header for {} ...done'.format(x))

    def test_segy_scantraces(self):
        """Scan and report SEGY first and last trace."""
        self.getlogger(sys._getframe(1).f_code.co_name)

        self.logger.info('Scan traces...')

        x = Cube().scan_segy_traces('../../testdata/Cube/G/nh0304.segy',
                                    outfile="TMP/cube_scantraces")

        self.logger.info(x)

    def test_segy_import(self):
        """Import SEGY using internal reader (case 1 Grane)."""
        self.getlogger(sys._getframe(1).f_code.co_name)

        self.logger.info('Import SEGY format')

        x = Cube()
        x.from_file('../../testdata/Cube/G/nh0304.segy',
                    fformat='segy')

        self.assertEqual(x.nx, 257, 'NX')

        self.logger.info('Import SEGY format done')

        dim = x.values.shape

        self.logger.info("Dimension is {}".format(dim))
        self.assertEqual(dim, (257, 1165, 251), 'Dimensions 3D')
        self.assertAlmostEqual(x.values[100, 1000, 10], 3.6744, places=3)

        self.logger.debug('XORI: {}'.format(x.xori))
        self.logger.debug('XINC: {}'.format(x.xinc))
        self.logger.debug('YORI: {}'.format(x.yori))
        self.logger.debug('YINC: {}'.format(x.yinc))
        self.logger.debug('ZORI: {}'.format(x.zori))
        self.logger.debug('ZINC: {}'.format(x.zinc))
        self.logger.debug('ROTA: {}'.format(x.rotation))
        self.logger.debug('YFLP: {}'.format(x.yflip))

    def test_segyio_import(self):
        """Import SEGY (case 1 Grane) via SegIO library."""
        self.getlogger(sys._getframe(1).f_code.co_name)

        self.logger.info('Import SEGY format via SEGYIO')

        x = Cube()
        x.from_file('../../testdata/Cube/G/nh0304.segy',
                    fformat='segy', engine=1)

        self.assertEqual(x.nx, 257, 'NX')
        dim = x.values.shape

        self.logger.info("Dimension is {}".format(dim))
        self.assertEqual(dim, (257, 1165, 251), 'Dimensions 3D')

        self.assertAlmostEqual(x.values[100, 1000, 10], 3.6744, places=3)

        self.logger.debug('XORI: {}'.format(x.xori))
        self.logger.debug('XINC: {}'.format(x.xinc))
        self.logger.debug('YORI: {}'.format(x.yori))
        self.logger.debug('YINC: {}'.format(x.yinc))
        self.logger.debug('ZORI: {}'.format(x.zori))
        self.logger.debug('ZINC: {}'.format(x.zinc))
        self.logger.debug('ROTA: {}'.format(x.rotation))
        self.logger.debug('YFLP: {}'.format(x.yflip))

        self.logger.info('Import SEGY format done')

    def test_more_segy_import(self):
        """Import SEGY (Gullfaks) via internal and SegIO library."""
        self.getlogger(sys._getframe(1).f_code.co_name)

        self.logger.info('Import Gullfaks SEGY format via SEGY and  SEGYIO')

        sfile = '../../testdata/Cube/G/nh0304.segy'

        x1 = Cube()
        x1.from_file(sfile, fformat='segy', engine=0)

        x2 = Cube()
        x2.from_file(sfile, fformat='segy', engine=1)

        self.assertEqual(x1.nx, x2.nx, 'NX')

    def test_cube_storm_import(self):
        """Import SEGY (case 2 Gullfaks)."""
        self.getlogger(sys._getframe(1).f_code.co_name)

        self.logger.info('Import on STORM format...')

        x = Cube()
        x.from_file('../../testdata/Cube/GF/gf_depth_1985_10_01.storm',
                    fformat='storm')

        self.assertEqual(x.nx, 501, 'NX')

    def test_swapaxes(self):
        """Import SEGY and swap axes."""
        self.getlogger(sys._getframe(1).f_code.co_name)

        self.logger.info('Import SEGY (test swapaxes)')

        x = Cube()
        x.from_file('../../testdata/Cube/GF/gf_depth_1985_10_01.segy',
                    fformat='segy')

        orig_nx = x.nx
        orig_ny = x.ny
        orig_rot = x.rotation

        self.logger.info("Swap axes... (orig rotation is {})".format(orig_rot))
        x.swapaxes()
        self.logger.info("Swap axes... done")

        self.assertEqual(x.nx, orig_ny, 'NX swapaxes')
        self.assertEqual(x.ny, orig_nx, 'NY swapaxes')
        self.assertAlmostEqual(x.rotation, 359.1995, places=3)
        print(x.rotation)

        x.to_file("TMP/cube_swapped.rmsreg", fformat="rms_regular")


if __name__ == '__main__':

    unittest.main()
