import os

import xtgeo
from xtgeo.common import XTGeoDialog
import test_common.test_xtg as tsetup

roxver = None
try:
    import roxar
    roxver = roxar.__version__
except ImportError:
    pass

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

td = xtg.tmpdir
testpath = xtg.testpath

# =============================================================================
# Do tests
# =============================================================================
proj = {}
proj['1.1'] = '../xtgeo-testdata-equinor/data/rmsprojects/reek.rms10.1.1'
proj['1.1.1'] = '../xtgeo-testdata-equinor/data/rmsprojects/reek.rms10.1.3'
proj['1.2.1'] = '../xtgeo-testdata-equinor/data/rmsprojects/reek.rms11.0.1'
proj['1.3'] = '../xtgeo-testdata-equinor/data/rmsprojects/reek.rms11.1.0'


@tsetup.skipunlessroxar
def test_getsurface():
    """Get a surface from a RMS project."""

    print(roxver)

    if not os.path.isdir(proj[roxver]):
        raise RuntimeError('RMS test project is missing for roxar version {}'
                           .format(roxver))

    logger.info('Simple case, reading a surface from a Horizons category')

    x = xtgeo.surface.RegularSurface()
    x.from_roxar(proj[roxver], 'TopUpperReek', 'DS_extracted')

    x.to_file(os.path.join(td, 'topupperreek_from_rms.gri'))

    tsetup.assert_equal(x.ncol, 99, 'NCOL of from RMS')

    tsetup.assert_almostequal(x.values.mean(), 1696.255599, 0.001)


@tsetup.skipunlessroxar
def test_getsurface_from_zones():
    """Get a surface from a RMS project, from the zones container."""

    if not os.path.isdir(proj[roxver]):
        raise RuntimeError('RMS test project is missing for roxar version {}'
                           .format(roxver))

    logger.info('Simple case, reading a surface from a Horizons category')

    # direct initiate an instance from Roxar import
    x = xtgeo.surface_from_roxar(proj[roxver], 'UpperReek', 'IS_calculated',
                                 stype='zones')

    x.to_file(os.path.join(td, 'upperreek_from_rms.gri'))

    tsetup.assert_equal(x.ncol, 99, 'NCOL of from RMS')

    # values from mean and stddev are read from statistics in RMS
    tsetup.assert_almostequal(x.values.mean(), 20.8205, 0.001)
    tsetup.assert_almostequal(x.values.std(), 1.7867, 0.001)
    print(x.values.std())

    # write to folder
    x.to_roxar(proj[roxver], 'UpperReek', 'IS_jriv', stype='zones')
