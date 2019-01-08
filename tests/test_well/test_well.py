# -*- coding: utf-8 -*-
from __future__ import division, absolute_import
from __future__ import print_function

import glob
from os.path import join

import pytest

from xtgeo.well import Well
from xtgeo.xyz import Polygons
from xtgeo.common import XTGeoDialog

import test_common.test_xtg as tsetup

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

td = xtg.tmpdir
testpath = xtg.testpath

# =========================================================================
# Do tests
# =========================================================================

wfile = join(testpath, 'wells/reek/1/OP_1.w')
wfiles = join(testpath, 'wells/reek/1/*')


@pytest.fixture()
def loadwell1():
    """Fixture for loading a well (pytest setup)"""
    logger.info('Load well 1')
    return Well(wfile)


def test_import(loadwell1):
    """Import well from file."""

    mywell = loadwell1

    logger.debug('True well name:', mywell.truewellname)
    tsetup.assert_equal(mywell.xpos, 461809.59, 'XPOS')
    tsetup.assert_equal(mywell.ypos, 5932990.36, 'YPOS')
    tsetup.assert_equal(mywell.wellname, 'OP_1', 'WNAME')

    logger.info(mywell.get_logtype('Zonelog'))
    logger.info(mywell.get_logrecord('Zonelog'))
    logger.info(mywell.lognames_all)
    logger.info(mywell.dataframe)

    # logger.info the numpy string of Poro...
    logger.info(type(mywell.dataframe['Poro'].values))


def test_import_export_many():
    """ Import and export many wells (test speed)"""

    logger.debug(wfiles)

    for filename in sorted(glob.glob(wfiles)):
        logger.info('Importing ' + filename)
        mywell = Well(filename)
        logger.info(mywell.nrow)
        logger.info(mywell.ncol)
        logger.info(mywell.lognames)

        wname = join(td, mywell.xwellname + '.w')
        logger.info('Exporting ' + wname)
        mywell.to_file(wname)


def test_shortwellname():
    """Test that shortwellname gives wanted result"""

    mywell = Well()

    mywell._wname = '31/2-A-14 2H'
    short = mywell.shortwellname

    assert short == 'A-142H'

    mywell._wname = '6412_2-A-14_2H'
    short = mywell.shortwellname

    assert short == 'A-142H'


# def test_import_as_rms_export_as_hdf5_many():
#     """ Import RMS and export as HDF5, many"""

#     logger.debug(wfiles)

#     wfile = td + "/mytest.h5"
#     for filename in glob.glob(wfiles):
#         logger.info("Importing " + filename)
#         mywell = Well(filename)
#         logger.info(mywell.nrow)
#         logger.info(mywell.ncol)
#         logger.info(mywell.lognames)

#         wname = td + "/" + mywell.xwellname + ".h5"
#         logger.info("Exporting " + wname)
#         mywell.to_file(wfile, fformat='hdf5')


def test_get_carr(loadwell1):
    """Get a C array pointer"""

    mywell = loadwell1

    dummy = mywell.get_carray('NOSUCH')

    tsetup.assert_equal(dummy, None, 'Wrong log name')

    cref = mywell.get_carray('X_UTME')

    xref = str(cref)
    swig = False
    if 'Swig' in xref and 'double' in xref:
        swig = True

    tsetup.assert_equal(swig, True, 'carray from log name, double')

    cref = mywell.get_carray('Zonelog')

    xref = str(cref)
    swig = False
    if 'Swig' in xref and 'int' in xref:
        swig = True

    tsetup.assert_equal(swig, True, 'carray from log name, int')


def test_make_hlen(loadwell1):
    """Create a hlen log."""

    mywell = loadwell1
    mywell.create_relative_hlen()

    logger.debug(mywell.dataframe)


def test_rescale_well(loadwell1):
    """Rescale (resample) a well to a finer increment"""

    mywell = loadwell1

    df1 = mywell.dataframe.copy()
    df1 = df1[(df1['Z_TVDSS'] > 1587) & (df1['Z_TVDSS'] < 1610)]

    mywell.to_file('test1.w')

    mywell.rescale(delta=0.2)

    df2 = mywell.dataframe.copy()
    df2 = df2[(df2['Z_TVDSS'] > 1587) & (df2['Z_TVDSS'] < 1610)]
    print('XXXXXXXXXXXXXXXXXXX', df1['Perm'].mean())
    print('XXXXXXXXXXXXXXXXXXX', df2['Perm'].mean())
    mywell.to_file('test2.w')


def test_fence(loadwell1):
    """Return a resampled fence."""

    mywell = Well(wfile)
    pline = mywell.get_fence_polyline(extend=10, tvdmin=1000)

    logger.debug(pline)


def test_fence_as_polygons(loadwell1):
    """Return a resampled fence as Polygons."""

    mywell = Well(wfile)
    pline = mywell.get_fence_polyline(extend=3, tvdmin=1000,
                                      asnumpy=False)

    assert isinstance(pline, Polygons)
    dfr = pline.dataframe
    print(dfr)
    tsetup.assert_almostequal(dfr['X_UTME'][5], 462567.741277, 0.0001)


def test_get_zonation_points():
    """Get zonations points (zone tops)"""

    mywell = Well(wfile, zonelogname='Zonelog')
    mywell.get_zonation_points()


def test_get_zone_interval():
    """Get zonations points (zone tops)"""

    mywell = Well(wfile, zonelogname='Zonelog')
    line = mywell.get_zone_interval(3)

    print(line)

    logger.info(type(line))

    tsetup.assert_almostequal(line.iat[0, 0], 462698.33299, 0.001)
    tsetup.assert_almostequal(line.iat[-1, 2], 1643.1618, 0.001)


# def test_get_zonation_holes():
#     """get a report of holes in the zonation, some samples with -999 """

#     mywell = Well(wfile, zonelogname='Zonelog')
#     report = mywell.report_zonation_holes()

#     logger.info('\n{}'.format(report))

#     tsetup.assert_equal(report.iat[0, 0], 4166)  # first value for INDEX
#     tsetup.assert_equal(report.iat[1, 3], 1570.3855)  # second value for Z


# def test_get_filled_dataframe():
#     """Get a filled DataFrame"""

#     mywell = Well(wfile)

#     df1 = mywell.dataframe

#     df2 = mywell.get_filled_dataframe()

#     logger.debug(df1)
#     logger.debug(df2)
