import os
import pytest
import numpy as np

from xtgeo.surface import RegularSurface
from xtgeo.common import XTGeoDialog
from xtgeo.xyz import Points
import test_common.test_xtg as tsetup

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

XTGSHOW = False
if 'XTG_SHOW' in os.environ:
    XTGSHOW = True

if not xtg.testsetup():
    raise SystemExit

TMPD = xtg.tmpdir

# =============================================================================
# Do tests
# =============================================================================
ftop1 = '../xtgeo-testdata/surfaces/reek/1/topreek_rota.gri'


@pytest.fixture()
def reek_map():
    logger.info('Loading surface')
    return RegularSurface(ftop1)


def test_resample(reek_map):
    """Do resampling from one surface to another"""

    xs = reek_map
    assert xs.ncol == 554

    xs_copy = xs.copy()

    # create a new map instance, unrotated, based on this map
    ncol = int((xs.xmax - xs.xmin) / 10)
    nrow = int((xs.ymax - xs.ymin) / 10)
    values = np.zeros((nrow, ncol))
    snew = RegularSurface(xori=xs.xmin, xinc=10, yori=xs.ymin, yinc=10,
                          nrow=nrow, ncol=ncol, values=values)

    snew.resample(xs)

    fout = os.path.join(TMPD, 'reek_resampled.gri')
    snew.to_file(fout, fformat='irap_binary')

    tsetup.assert_almostequal(snew.values.mean(), 1698.458, 2)
    tsetup.assert_almostequal(snew.values.mean(), xs.values.mean(), 2)

    # check that the "other" in snew.resample(other) is unchanged:
    assert xs.xinc == xs_copy.xinc
    tsetup.assert_almostequal(xs.values.mean(), xs_copy.values.mean(), 1e-4)
    tsetup.assert_almostequal(xs.values.std(), xs_copy.values.std(), 1e-4)


def test_refine(reek_map):
    """Do refining of a surface"""

    xs = reek_map
    assert xs.ncol == 554

    xs_orig = xs.copy()
    xs.refine(4)

    fout = os.path.join(TMPD, 'reek_refined.gri')
    xs.to_file(fout, fformat='irap_binary')

    tsetup.assert_almostequal(xs_orig.values.mean(), xs.values.mean(), 0.8)

    if XTGSHOW:
        logger.info('Output plots to file (may be time consuming)')
        xs_orig.quickplot(filename=os.path.join(TMPD, 'reek_orig.png'))
        xs.quickplot(filename=os.path.join(TMPD, 'reek_refined4.png'))


def test_coarsen(reek_map):
    """Do a coarsening of a surface"""

    xs = reek_map
    assert xs.ncol == 554

    xs_orig = xs.copy()
    xs.coarsen(3)

    fout = os.path.join(TMPD, 'reek_coarsened.gri')
    xs.to_file(fout, fformat='irap_binary')

    tsetup.assert_almostequal(xs_orig.values.mean(), xs.values.mean(), 0.8)

    if XTGSHOW:
        logger.info('Output plots to file (may be time consuming)')
        xs_orig.quickplot(filename=os.path.join(TMPD, 'reek_orig.png'))
        xs.quickplot(filename=os.path.join(TMPD, 'reek_coarsen3.png'))


@tsetup.bigtest
def test_points_gridding(reek_map):
    """Make points of surface; then grid back to surface."""

    xs = reek_map
    assert xs.ncol == 554

    xyz = Points(xs)

    xyz.dataframe['Z_TVDSS'] = xyz.dataframe['Z_TVDSS'] + 300

    logger.info('Avg of points: {}'.format(xyz.dataframe['Z_TVDSS'].mean()))

    xscopy = xs.copy()

    logger.info(xs.values.flags)
    logger.info(xscopy.values.flags)

    # now regrid
    xscopy.gridding(xyz, coarsen=1)  # coarsen will speed up test a lot

    if XTGSHOW:
        logger.info('Output plots to file (may be time consuming)')
        xs.quickplot(filename=os.path.join(TMPD, 's1.png'))
        xscopy.quickplot(filename=os.path.join(TMPD, '/tmp/s2.png'))

    tsetup.assert_almostequal(xscopy.values.mean(), xs.values.mean() + 300, 2)

    xscopy.to_file(os.path.join(TMPD, 'reek_points_to_map.gri'),
                   fformat='irap_binary')
