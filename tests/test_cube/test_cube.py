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
sfile3 = testpath + '/cubes/reek/syntseis_20000101_seismic_depth_stack.storm'
sfile4 = testpath + '/cubes/etc/testx.segy'


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


# def test_segy_import_cvalues():
#     """Import SEGY using internal reader (case 1 Reek) and chk issues."""

#     logger.info('Import SEGY format')

#     x = Cube()

#     x.from_file(sfile1, fformat='segy')

#     np1 = x.values.copy()

#     np2 = x.values

#     assert np.array_equal(np1, np2)

def test_storm_import():
    """Import Cube using Storm format (case Reek)."""

    logger.info('Import SEGY format')

    acube = Cube()

    st1 = xtg.timer()
    acube.from_file(sfile3, fformat='storm')
    elapsed = xtg.timer(st1)
    logger.info('Reading Storm format took {}'.format(elapsed))

    assert acube.ncol == 280, 'NCOL'

    vals = acube.values

    tsetup.assert_almostequal(vals[180, 185, 4], 0.117074, 0.0001)

    acube.to_file('TMP/cube.rmsreg', fformat='rms_regular')


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


def test_segyio_import_export_pristine():
    """Import and export as pristine SEGY (case 1 Reek) via SegIO library."""

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

    x.to_file('TMP/reek_cube_pristine.segy', pristine=True)

    # # reread that file
    # y = Cube('TMP/reek_cube_pristine.segy')

    # logger.info(y.values.mean())


def test_segyio_export_xtgeo():
    """Import via SEGYIO and and export SEGY (case 1 Reek) via XTGeo."""

    logger.info('Import SEGY format via SEGYIO')

    x = Cube(sfile1)

    x.values += 200

    x.to_file('TMP/reek_cube_xtgeo.segy', engine='xtgeo')

    xx = Cube()
    xx.scan_segy_header('TMP/reek_cube_xtgeo.segy',
                        outfile='TMP/cube_scanheader2')
    xx.scan_segy_traces('TMP/reek_cube_xtgeo.segy',
                        outfile='TMP/cube_scantraces2')

    # # reread that file, scan header
    # y = Cube('TMP/reek_cube_pristine.segy')

    # logger.info(y.values.mean())


def test_cube_resampling():
    """Import a cube, then make a smaller and resample, then export the new"""

    logger.info('Import SEGY format via SEGYIO')

    incube = Cube(sfile1)

    newcube = Cube(xori=460500, yori=5926100, zori=1540,
                   xinc=40, yinc=40, zinc=5, ncol=200, nrow=100,
                   nlay=100, rotation=incube.rotation, yflip=incube.yflip)

    newcube.resample(incube, sampling='trilinear', outside_value=10.0)

    tsetup.assert_almostequal(newcube.values.mean(), 5.3107, 0.0001)
    tsetup.assert_almostequal(newcube.values[20, 20, 20], 10.0, 0.0001)

    newcube.to_file('TMP/cube_resmaple1.segy')


def test_cube_thinning():
    """Import a cube, then make a smaller by thinning every N line"""

    logger.info('Import SEGY format via SEGYIO')

    incube = Cube(sfile1)
    incube.describe()

    # thinning to evey second column and row, but not vertically
    incube.do_thinning(2, 2, 1)
    incube.describe()

    incube.to_file('TMP/cube_thinned.segy')

    incube2 = Cube('TMP/cube_thinned.segy')
    incube2.describe()


def test_cube_cropping():
    """Import a cube, then make a smaller by cropping"""

    logger.info('Import SEGY format via SEGYIO')

    incube = Cube(sfile1)

    # thinning to evey second column and row, but not vertically
    incube.do_cropping((2, 13), (10, 22), (30, 0))

    incube.to_file('TMP/cube_cropped.segy')


def test_cube_swapaxes():
    """Import a cube, do axes swapping back and forth"""

    logger.info('Import SEGY format via SEGYIO')

    incube = Cube(sfile4)
    incube.describe()
    val1 = incube.values.copy()

    incube.swapaxes()
    incube.describe()

    incube.swapaxes()
    val2 = incube.values.copy()
    incube.describe()

    diff = val1 - val2

    tsetup.assert_almostequal(diff.mean(), 0.0, 0.000001)
    tsetup.assert_almostequal(diff.std(), 0.0, 0.000001)
    assert incube.ilines.size == incube.ncol
