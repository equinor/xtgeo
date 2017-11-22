import numpy as np
import os
import os.path
import sys

from xtgeo.surface import RegularSurface
from xtgeo.common import XTGeoDialog
from .test_xtg import assert_equal, assert_almostequal

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg._testsetup():
    sys.exit(-9)

td = xtg.tmpdir
testpath = xtg.testpath


# =============================================================================
# Do tests
# =============================================================================


def test_create():
    """Create default surface"""

    logger.info('Simple case...')

    x = RegularSurface()
    assert_equal(x.ncol, 5, 'NX')
    assert_equal(x.nrow, 3, 'NY')
    v = x.values
    xdim, ydim = v.shape
    assert_equal(xdim, 5, 'NX from DIM')


def test_irapasc_export():
    """Export Irap ASCII (1)."""

    logger.info('Export to Irap Classic')
    x = RegularSurface()
    x.to_file('TMP/irap.fgr', fformat="irap_ascii")

    fstatus = False
    if os.path.isfile('TMP/irap.fgr') is True:
        fstatus = True

    assert fstatus is True


def test_irapasc_exp2():
    """Export Irap ASCII (2)."""

    logger.info('Export to Irap Classic and Binary')

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
    assert_equal(x.ncol, 120)
    x.to_file('TMP/irap2_a.fgr', fformat="irap_ascii")
    x.to_file('TMP/irap2_b.gri', fformat="irap_binary")

    fsize = os.path.getsize('TMP/irap2_b.gri')
    logger.info(fsize)
    assert_equal(fsize, 48900)


def test_minmax_rotated_map():
    """Min and max of rotated map"""
    logger.info('Import and export...')

    x = RegularSurface()
    x.from_file('../xtgeo-testdata/surfaces/fos/1/fossekall1.irapbin',
                fformat='irap_binary')

    assert_almostequal(x.xmin, 464308.40625, 0.01)
    assert_almostequal(x.xmax, 466017.38536, 0.01)
    assert_almostequal(x.ymin, 7335894.4380, 0.01)
    assert_almostequal(x.ymax, 7337678.1262, 0.01)


def test_irapbin_io():
    """Import and export Irap binary."""
    logger.info('Import and export...')

    x = RegularSurface()
    x.from_file('../xtgeo-testdata/surfaces/fos/1/fossekall1.irapbin',
                fformat='irap_binary')

    x.to_file('TMP/foss1_test.fgr', fformat='irap_ascii')

    logger.debug("NX is {}".format(x.ncol))

    assert_equal(x.ncol, 58)

    # get the 1D numpy
    v1d = x.get_zval()

    logger.info('Mean VALUES are: {}'.format(np.nanmean(v1d)))

    zval = x.values

    logger.info('VALUES are:\n{}'.format(zval))

    logger.info('MEAN value (original):\n{}'.format(zval.mean()))

    # add value via numpy
    zval = zval + 300
    # update
    x.values = zval

    logger.info('MEAN value (update):\n{}'.format(x.values.mean()))

    assert_almostequal(x.values.mean(), 2882.741, 0.01)

    x.to_file('TMP/foss1_plus_300_a.fgr', fformat='irap_ascii')
    x.to_file('TMP/foss1_plus_300_b.gri', fformat='irap_binary')

    mfile = '../xtgeo-testdata/surfaces/fos/1/fossekall1.irapbin'

    # direct import
    y = RegularSurface(mfile)
    assert_equal(y.ncol, 58)

    # semidirect import
    cc = RegularSurface().from_file(mfile)
    assert_equal(cc.ncol, 58)


def test_get_xy_value_lists():
    """Get the xy list and value list"""

    x = RegularSurface()
    x.from_file('../xtgeo-testdata/surfaces/fos/1/fossekall1.irapbin',
                fformat='irap_binary')

    xylist, valuelist = x.get_xy_value_lists(valuefmt='8.3f',
                                             xyfmt='12.2f')

    logger.info(xylist[2])
    logger.info(valuelist[2])

    assert_equal(valuelist[2], 2813.981)


def test_similarity():
    """Testing similarity of two surfaces. 0.0 means identical in
    terms of mean value.
    """

    logger.info('Test if surfaces are similar...')

    mfile = '../xtgeo-testdata/surfaces/fos/1/fossekall1.irapbin'

    x = RegularSurface(mfile)
    y = RegularSurface(mfile)

    si = x.similarity_index(y)
    assert_equal(si, 0.0)

    y.values = y.values * 2

    si = x.similarity_index(y)
    assert_equal(si, 1.0)


def test_irapbin_io_loop():
    """Do a loop over big Troll data set."""

    n = 10
    logger.info("Import and export map to numpy {} times".format(n))

    for i in range(0, 10):
        # print(i)
        x = RegularSurface()
        x.from_file('../xtgeo-testdata/surfaces/tro/1/troll.irapbin',
                    fformat='irap_binary')

        logger.info('Map dimensions: {} {}'.format(x.ncol, x.nrow))

        m1 = x.values.mean()
        zval = x.values
        zval = zval + 300
        x.values = zval
        m2 = x.values.mean()
        x.to_file('TMP/troll.gri', fformat='irap_binary')
        logger.info("Mean before and after: {} .. {}".format(m1, m2))

#     xtg.info("Import and export map to numpy {} times DONE".format(n))


def test_distance_from_point():
    """Distance from point."""

    x = RegularSurface()
    x.from_file('../xtgeo-testdata/surfaces/fos/1/fossekall1.irapbin',
                fformat='irap_binary')

    x.distance_from_point(point=(464960, 7336900), azimuth=30)

    x.to_file('TMP/foss1_dist_point.gri', fformat='irap_binary')


def test_value_from_xy():
    """
    get Z value from XY point
    """

    x = RegularSurface()
    x.from_file('../xtgeo-testdata/surfaces/fos/1/fossekall1.irapbin',
                fformat='irap_binary')

    z = x.get_value_from_xy(point=(464375.992, 7337128.076))

    z = float("{:6.2f}".format(z))

    assert_equal(z, 2766.96)

    # outside value shall return None
    z = x.get_value_from_xy(point=(0.0, 7337128.076))
    assert z is None


def test_fence():
    """
    Test sampling a fence from a surface.
    """

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

    logger.debug("NP:")
    logger.debug(myfence)

    x = RegularSurface()
    x.from_file('../xtgeo-testdata/surfaces/fos/1/fossekall1.irapbin',
                fformat='irap_binary')

    newfence = x.get_fence(myfence)

    logger.debug("updated NP:")
    logger.debug(newfence)
