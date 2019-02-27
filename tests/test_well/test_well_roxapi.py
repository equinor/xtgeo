import os
from os.path import join

import xtgeo
from xtgeo.common import XTGeoDialog
import test_common.test_xtg as tsetup

roxv = None
try:
    import roxar
    roxv = roxar.__version__
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
proj['1.2.1'] = '../xtgeo-testdata-equinor/data/rmsprojects/reek.rms11.0.1'
proj['1.3'] = '../xtgeo-testdata-equinor/data/rmsprojects/reek.rms11.1.0'


@tsetup.skipunlessroxar
def test_getwell():
    """Get a well from a RMS project."""

    print(roxv)

    if not os.path.isdir(proj[roxv]):
        raise RuntimeError('RMS test project is missing for roxar version {}'
                           .format(roxv))

    logger.info('Simple case, reading a well from RMS well folder')

    xwell = xtgeo.well.Well()
    xwell.from_roxar(proj[roxv], 'WI_3_RKB2', trajectory='Drilled trajectory',
                     logrun='LOG', lognames=['Zonelog', 'Poro', 'Facies'])

    logger.info('Dataframe\n %s ', xwell.dataframe)

    tsetup.assert_equal(xwell.nrow, 10081, 'NROW of well')
    tsetup.assert_equal(xwell.rkb, -10, 'RKB of well')

    xwell.to_file(join(td, 'roxwell_export.rmswell'))

    # tsetup.assert_almostequal(x.values.mean(), 1696.255599, 0.001)


@tsetup.skipunlessroxar
def test_getwell_all_logs():
    """Get a well from a RMS project, reading all logs present."""

    print(roxv)

    if not os.path.isdir(proj[roxv]):
        raise RuntimeError('RMS test project is missing for roxar version {}'
                           .format(roxv))

    logger.info('Simple case, reading a well from RMS well folder')

    xwell = xtgeo.well.Well()
    xwell.from_roxar(proj[roxv], 'WI_3_RKB2',
                     trajectory='Drilled trajectory',
                     logrun='LOG', lognames='all')

    logger.info('Dataframe\n %s ', xwell.dataframe)

    tsetup.assert_equal(xwell.nrow, 10081, 'NROW of well')
    tsetup.assert_equal(xwell.rkb, -10, 'RKB of well')

    xwell.to_file(join(td, 'roxwell_export.rmswell'))

    # tsetup.assert_almostequal(x.values.mean(), 1696.255599, 0.001)
