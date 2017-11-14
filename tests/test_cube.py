# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pytest

from xtgeo.cube import Cube
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg._testsetup():
    sys.exit(-9)

td = xtg.tmpdir
testpath = xtg.testpath

skiplargetest = pytest.mark.skipif(xtg.bigtest is False,
                                   reason="Big tests skip")

# =============================================================================
# Do tests
# =============================================================================

sfile1 = testpath + '/cubes/gra/nh0304.segy'
sfile2 = testpath + '/cubes/gfb/gf_depth_1985_10_01.segy'


@skiplargetest
def test_create():
    """Create default cube instance."""
    x = Cube()
    assert x.ncol == 5, 'NCOL'
    assert x.nrow == 3, 'NROW'
    v = x.values
    xdim, ydim, zdim = v.shape
    assert xdim == 5, 'NX from numpy shape '


def test_segy_scanheader():
    """Scan SEGY and report header, using XTGeo internal reader."""
    logger.info('Scan header...')

    if not os.path.isfile(sfile1):
        raise Exception("No such file")

    x = Cube().scan_segy_header(sfile1, outfile="TMP/cube_scanheader")

    logger.info('Scan header for {} ...done'.format(x))


def test_segy_scantraces():
    """Scan and report SEGY first and last trace (internal reader)."""

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

    logger.info('Import SEGY format')

    x = Cube()

    x.from_file(sfile1, fformat='segy')

    np1 = x.values.copy()

    np2 = x.values

    assert np.array_equal(np1, np2)


@skiplargetest
def test_segy_import():
    """Import SEGY using internal reader (case 1 Grane)."""

    logger.info('Import SEGY format')

    x = Cube()

    x.from_file(sfile1, fformat='segy')

    assert x.ncol == 257, 'NCOL'

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

    logger.info('Import SEGY format via SEGYIO')

    x = Cube()
    x.from_file(sfile1, fformat='segy', engine='segyio')

    assert x.ncol == 257, 'NCOL'
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
#         x1.from_file(sfile1, fformat='segy', engine='xtgeo')

#         x2 = Cube()
#         x2.from_file(sfile1, fformat='segy', engine='segyio')

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
