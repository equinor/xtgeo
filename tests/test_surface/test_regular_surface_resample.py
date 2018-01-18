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
ftop1 = '../xtgeo-testdata/surfaces/fos/2/topaare1.gri'


@pytest.fixture()
def fos_map():
    logger.info('Loading surface')
    return RegularSurface(ftop1)


def test_resample(fos_map):
    """Do resampling from one surface to another"""

    xs = fos_map
    assert xs.ncol == 97

    # create a new map instance, unrotated, based on this map
    ncol = int((xs.xmax - xs.xmin) / 10)
    nrow = int((xs.ymax - xs.ymin) / 10)
    values = np.zeros((nrow, ncol), order='F')
    snew = RegularSurface(xori=xs.xmin, xinc=10, yori=xs.ymin, yinc=10,
                          nrow=nrow, ncol=ncol, values=values)

    print(snew)

    snew.resample(xs)

    fout = os.path.join(td, 'fos_resampled.gri')
    print('EXPORT', fout)
    snew.to_file(fout, fformat='irap_binary')

    tsetup.assert_almostequal(snew.values.mean(), 2791.21, 2)
    tsetup.assert_almostequal(snew.values.mean(), xs.values.mean(), 2)


def test_points_gridding(fos_map):
    """Make points of surface; then grid back to surface."""

    xs = fos_map
    assert xs.ncol == 97

    xyz = Points(xs)

    print(xyz.dataframe)

    xyz.dataframe['Z'] = xyz.dataframe['Z'] + 300

    print(xyz.dataframe)

    xscopy = xs.copy()

    # now regrid
    xscopy.gridding(xyz)

    tsetup.assert_almostequal(xscopy.values.mean(), xs.values.mean() + 300, 2)

    xscopy.to_file(os.path.join(td, 'fos_points_to_map.gri'),
                   fformat='irap_binary')
