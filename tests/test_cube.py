import os
import logging
import sys
import numpy as np
import pytest

from xtgeo.cube import Cube
from xtgeo.common import XTGeoDialog


testpath = os.getenv('XTGEO_TESTDATA', '../xtgeo-testdata')

path = 'TMP'
try:
    os.makedirs(path)
except OSError:
    if not os.path.isdir(path):
        raise

xtg = XTGeoDialog()


if 'XTGEO_NO_BIGTESTS' in os.environ:
    bigtest = False
else:
    bigtest = True

skiplargetest = pytest.mark.skipif(bigtest is False, reason="Big tests skip")

# =============================================================================
# Do tests
# =============================================================================

sfile1 = testpath + '/cubes/gra/nh0304.segy'
sfile2 = testpath + '/cubes/gfb/gf_depth_1985_10_01.segy'


def getlogger(name):

    format = xtg.loggingformat

    logging.basicConfig(format=format, stream=sys.stdout)
    logging.getLogger().setLevel(xtg.logginglevel)  # root logger!

    return logging.getLogger(name)


@skiplargetest
def test_create():
    """Create default cube instance."""
    x = Cube()
    assert x.nx == 5, 'NX'
    assert x.ny == 3, 'NY'
    v = x.values
    xdim, ydim, zdim = v.shape
    assert xdim == 5, 'NX from numpy shape '


def test_segy_scanheader():
    """Scan SEGY and report header, using XTGeo internal reader."""
    logger = getlogger(sys._getframe(1).f_code.co_name)
    logger.info('Scan header...')

    if not os.path.isfile(sfile1):
        raise Exception("No such file")

    x = Cube().scan_segy_header(sfile1, outfile="TMP/cube_scanheader")

    logger.info('Scan header for {} ...done'.format(x))


def test_segy_scantraces():
    """Scan and report SEGY first and last trace (internal reader)."""
    logger = getlogger(sys._getframe(1).f_code.co_name)

    logger.info('Scan traces...')

    x = Cube().scan_segy_traces(sfile1, outfile="TMP/cube_scantraces")

    with open("TMP/cube_scantraces") as lines:
        for line in lines:
            print(line)

    # line = open('TMP/cube_scantraces', 'r').readlines()[0]
    # logger.info(x)
    # print line


def test_segy_import_cvalues():
    """Import SEGY using internal reader (case 1 Grane) and chk issues."""
    logger = getlogger(sys._getframe(1).f_code.co_name)

    logger.info('Import SEGY format')

    x = Cube()

    x.from_file(sfile1, fformat='segy')

    np1 = x.values.copy()

    np2 = x.values

    assert np.array_equal(np1, np2)


@skiplargetest
def test_segy_import():
    """Import SEGY using internal reader (case 1 Grane)."""
    logger = getlogger(sys._getframe(1).f_code.co_name)

    logger.info('Import SEGY format')

    x = Cube()

    x.from_file(sfile1, fformat='segy')

    assert x.nx == 257, 'NX'

    logger.info('Import SEGY format done')

    dim = x.values.shape

    logger.info("Dimension is {}".format(dim))
    assert dim == (257, 1165, 251), 'Dimensions 3D'
    assert abs(x.values[100, 1000, 10] - 3.6744) < 0.001

    logger.debug('XORI: {}'.format(x.xori))
    logger.debug('XINC: {}'.format(x.xinc))
    logger.debug('YORI: {}'.format(x.yori))
    logger.debug('YINC: {}'.format(x.yinc))
    logger.debug('ZORI: {}'.format(x.zori))
    logger.debug('ZINC: {}'.format(x.zinc))
    logger.debug('ROTA: {}'.format(x.rotation))
    logger.debug('YFLP: {}'.format(x.yflip))


def test_segyio_import():
    """Import SEGY (case 1 Grane) via SegIO library."""
    logger = getlogger(sys._getframe(1).f_code.co_name)

    logger.info('Import SEGY format via SEGYIO')

    x = Cube()
    x.from_file(sfile1, fformat='segy', engine=1)

    assert x.nx == 257, 'NX'
    dim = x.values.shape

    logger.info("Dimension is {}".format(dim))
    assert dim == (257, 1165, 251), 'Dimensions 3D'

    assert abs(x.values[100, 1000, 10] - 3.6744) < 0.0001

    logger.debug('XORI: {}'.format(x.xori))
    logger.debug('XINC: {}'.format(x.xinc))
    logger.debug('YORI: {}'.format(x.yori))
    logger.debug('YINC: {}'.format(x.yinc))
    logger.debug('ZORI: {}'.format(x.zori))
    logger.debug('ZINC: {}'.format(x.zinc))
    logger.debug('ROTA: {}'.format(x.rotation))
    logger.debug('YFLP: {}'.format(x.yflip))

    logger.info('Import SEGY format done')

#     def test_more_segy_import(self):
#         """Import SEGY (Gullfaks) via internal and SegIO library."""
#         self.getlogger(sys._getframe(1).f_code.co_name)

#         self.logger.info('Import Gullfaks SEGY format via SEGY and  SEGYIO')

#         x1 = Cube()
#         x1.from_file(sfile1, fformat='segy', engine=0)

#         x2 = Cube()
#         x2.from_file(sfile1, fformat='segy', engine=1)

#         self.assertEqual(x1.nx, x2.nx, 'NX')

#     # def test_cube_storm_import(self):
#     #     """Import SEGY (case 2 Gullfaks)."""
#     #     self.getlogger(sys._getframe(1).f_code.co_name)

#     #     self.logger.info('Import on STORM format...')

#     #     x = Cube()
#     #     x.from_file('../../testdata/Cube/GF/gf_depth_1985_10_01.storm',
#     #                 fformat='storm')

#     #     self.assertEqual(x.nx, 501, 'NX')

#     def test_swapaxes(self):
#         """Import SEGY and swap axes."""
#         self.getlogger(sys._getframe(1).f_code.co_name)

#         self.logger.info('Import SEGY (test swapaxes)')

#         x = Cube()
#         x.from_file(sfile2, fformat='segy')

#         orig_nx = x.nx
#         orig_ny = x.ny
#         orig_rot = x.rotation

#         self.logger.info("Swap axes... (orig rotation is {})".format(orig_rot))
#         x.swapaxes()
#         self.logger.info("Swap axes... done")

#         self.assertEqual(x.nx, orig_ny, 'NX swapaxes')
#         self.assertEqual(x.ny, orig_nx, 'NY swapaxes')
#         self.assertAlmostEqual(x.rotation, 359.1995, places=3)
#         self.logger.info(x.rotation)

#         x.to_file("TMP/cube_swapped.rmsreg", fformat="rms_regular")
