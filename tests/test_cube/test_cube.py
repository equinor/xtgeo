# -*- coding: utf-8 -*-
import os
import sys
import numpy as np

from xtgeo.cube import Cube
from xtgeo.common import XTGeoDialog

import test_common.test_xtg as tsetup

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    sys.exit(-9)

td = xtg.tmpdir
testpath = xtg.testpath

# skipsegyio = pytest.mark.skipif(sys.version_info > (2, 7),
#                                 reason='Skip test with segyio for ver 3')

# =============================================================================
# Do tests
# =============================================================================

sfile1 = testpath + '/cubes/reek/syntseis_20000101_seismic_depth_stack.segy'
sfile2 = testpath + '/cubes/reek/syntseis_20030101_seismic_depth_stack.segy'


@tsetup.skipifroxar
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
        raise Exception('No such file')

    x = Cube().scan_segy_header(sfile1, outfile='TMP/cube_scanheader')

    logger.info('Scan header for {} ...done'.format(x))


def test_segy_scantraces():
    """Scan and report SEGY first and last trace (internal reader)."""

    logger.info('Scan traces...')

    x = Cube().scan_segy_traces(sfile1, outfile='TMP/cube_scantraces')

    logger.info('Object is {}'.format(x))
    with open('TMP/cube_scantraces') as lines:
        for line in lines:
            print(line)

    # line = open('TMP/cube_scantraces', 'r').readlines()[0]
    # logger.info(x)
    # print line


def test_segy_import_cvalues():
    """Import SEGY using internal reader (case 1 Reek) and chk issues."""

    logger.info('Import SEGY format')

    x = Cube()

    x.from_file(sfile1, fformat='segy')

    np1 = x.values.copy()

    np2 = x.values

    assert np.array_equal(np1, np2)


# @skipsegyio
# @skiplargetest
def test_segy_import():
    """Import SEGY using internal reader (case 1 Reek)."""

    logger.info('Import SEGY format')

    x = Cube()

    st1 = xtg.timer()
    x.from_file(sfile1, fformat='segy')
    elapsed = xtg.timer(st1)
    logger.info('Reading with XTGEO took {}'.format(elapsed))

    assert x.ncol == 408, 'NCOL'

    logger.info('Import SEGY format done')

    dim = x.values.shape

    logger.info('Dimension is {}'.format(dim))
    assert dim == (408, 280, 70), 'Dimensions 3D'

    print(x.values.max())
    tsetup.assert_almostequal(x.values.max(), 7.42017, 0.001)

    logger.debug('XORI: {}'.format(x.xori))
    logger.debug('XINC: {}'.format(x.xinc))
    logger.debug('YORI: {}'.format(x.yori))
    logger.debug('YINC: {}'.format(x.yinc))
    logger.debug('ZORI: {}'.format(x.zori))
    logger.debug('ZINC: {}'.format(x.zinc))
    logger.debug('ROTA: {}'.format(x.rotation))
    logger.debug('YFLP: {}'.format(x.yflip))


@tsetup.skipsegyio
def test_segyio_import():
    """Import SEGY (case 1 Reek) via SegIO library."""

    logger.info('Import SEGY format via SEGYIO')

    x = Cube()
    st1 = xtg.timer()
    x.from_file(sfile1, fformat='segy', engine='segyio')
    elapsed = xtg.timer(st1)
    logger.info('Reading with SEGYIO took {}'.format(elapsed))

    assert x.ncol == 408, 'NCOL'
    dim = x.values.shape

    logger.info('Dimension is {}'.format(dim))
    assert dim == (408, 280, 70), 'Dimensions 3D'
    tsetup.assert_almostequal(x.values.max(), 7.42017, 0.001)

    logger.debug('XORI: {}'.format(x.xori))
    logger.debug('XINC: {}'.format(x.xinc))
    logger.debug('YORI: {}'.format(x.yori))
    logger.debug('YINC: {}'.format(x.yinc))
    logger.debug('ZORI: {}'.format(x.zori))
    logger.debug('ZINC: {}'.format(x.zinc))
    logger.debug('ROTA: {}'.format(x.rotation))
    logger.debug('YFLP: {}'.format(x.yflip))

    logger.info('Import SEGY format done')


@tsetup.skipsegyio
def test_segyio_import_export():
    """Import and export SEGY (case 1 Reek) via SegIO library."""

    logger.info('Import SEGY format via SEGYIO')

    x = Cube()
    x.from_file(sfile1, fformat='segy', engine='segyio')

    assert x.ncol == 408, 'NCOL'
    dim = x.values.shape

    logger.info('Dimension is {}'.format(dim))
    assert dim == (408, 280, 70), 'Dimensions 3D'
    tsetup.assert_almostequal(x.values.max(), 7.42017, 0.001)

    input_mean = x.values.mean()

    logger.info(input_mean)

    x.values += 200

    x.to_file('TMP/reek_cube.segy')

    # reread that file
    y = Cube('TMP/reek_cube.segy')

    logger.info(y.values.mean())
