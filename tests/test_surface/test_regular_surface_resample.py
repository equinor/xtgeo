import os
import pytest
import numpy as np

from xtgeo.surface import RegularSurface
from xtgeo.common import XTGeoDialog
from xtgeo.xyz import Points
import test_common.test_xtg as tsetup

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

td = xtg.tmpdir

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

    # create a new map instance, unrotated, based on this map
    ncol = int((xs.xmax - xs.xmin) / 10)
    nrow = int((xs.ymax - xs.ymin) / 10)
    values = np.zeros((nrow, ncol))
    snew = RegularSurface(xori=xs.xmin, xinc=10, yori=xs.ymin, yinc=10,
                          nrow=nrow, ncol=ncol, values=values)

    snew.resample(xs)

    fout = os.path.join(td, 'reek_resampled.gri')
    snew.to_file(fout, fformat='irap_binary')

    tsetup.assert_almostequal(snew.values.mean(), 1698.458, 2)
    tsetup.assert_almostequal(snew.values.mean(), xs.values.mean(), 2)


def test_refine(reek_map):
    """Do refining of a surface"""

    xs = reek_map
    assert xs.ncol == 554

    xs_orig = xs.copy()
    xs.refine(4)

    fout = os.path.join(td, 'reek_refined.gri')
    xs.to_file(fout, fformat='irap_binary')

    tsetup.assert_almostequal(xs_orig.values.mean(), xs.values.mean(), 0.8)


def test_points_gridding(reek_map):
    """Make points of surface; then grid back to surface."""

    xs = reek_map
    assert xs.ncol == 554

    xyz = Points(xs)

    xyz.dataframe['Z_TVDSS'] = xyz.dataframe['Z_TVDSS'] + 300

    logger.info('Avg of points: {}'.format(xyz.dataframe['Z_TVDSS'].mean()))

    xscopy = xs.copy()

    print(xs.values.flags)
    print(xscopy.values.flags)

    # now regrid
    xscopy.gridding(xyz, coarsen=1)  # coarsen will speed up test a lot

    xs.quickplot(filename='/tmp/s1.png')
    xscopy.quickplot(filename='/tmp/s2.png')

    tsetup.assert_almostequal(xscopy.values.mean(), xs.values.mean() + 300, 2)

    xscopy.to_file(os.path.join(td, 'reek_points_to_map.gri'),
                   fformat='irap_binary')
