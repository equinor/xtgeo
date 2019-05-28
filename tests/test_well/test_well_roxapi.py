import os
from os.path import join

import pytest

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

TMPD = xtg.tmpdir
testpath = xtg.testpath

# =============================================================================
# Do tests
# =============================================================================
PROJ = {}
PROJ['1.1'] = '../xtgeo-testdata-equinor/data/rmsprojects/reek.rms10.1.1'
PROJ['1.1.1'] = '../xtgeo-testdata-equinor/data/rmsprojects/reek.rms10.1.3'
PROJ['1.2.1'] = '../xtgeo-testdata-equinor/data/rmsprojects/reek.rms11.0.1'
PROJ['1.3'] = '../xtgeo-testdata-equinor/data/rmsprojects/reek.rms11.1.0'

BPROJ = {}
BPROJ['1.3'] = '../xtgeo-testdata-equinor/data/rmsprojects/gfb2.rms11.1.0'


@tsetup.skipunlessroxar
def test_getwell():
    """Get a well from a RMS project."""

    print(roxv)

    if not os.path.isdir(PROJ[roxv]):
        raise RuntimeError('RMS test project is missing for roxar version {}'
                           .format(roxv))

    logger.info('Simple case, reading a well from RMS well folder')

    xwell = xtgeo.well.Well()
    xwell.from_roxar(PROJ[roxv], 'WI_3_RKB2', trajectory='Drilled trajectory',
                     logrun='LOG', lognames=['Zonelog', 'Poro', 'Facies'])

    logger.info('Dataframe\n %s ', xwell.dataframe)

    tsetup.assert_equal(xwell.nrow, 10081, 'NROW of well')
    tsetup.assert_equal(xwell.rkb, -10, 'RKB of well')

    df = xwell.dataframe

    tsetup.assert_almostequal(df.Poro.mean(), 0.191911, 0.001)

    xwell.to_file(join(TMPD, 'roxwell_export.rmswell'))


@tsetup.skipunlessroxar
def test_getwell_strict_logs_raise_error():
    """Get a well from a RMS project, with strict lognames, and this should fail"""

    if not os.path.isdir(PROJ[roxv]):
        raise RuntimeError('RMS test project is missing for roxar version {}'
                           .format(roxv))

    logger.info('Simple case, reading a well from RMS well folder')

    xwell = xtgeo.well.Well()

    rox = xtgeo.RoxUtils(PROJ[roxv])

    with pytest.raises(ValueError) as msg:
        logger.warning(msg)
        xwell.from_roxar(rox.project, 'WI_3_RKB2',
                         trajectory='Drilled trajectory',
                         logrun='LOG', lognames=['Zonelog', 'Poro', 'Facies', 'Dummy'],
                         lognames_strict=True)
    rox.safe_close()


@tsetup.skipunlessroxar
def test_getwell_all_logs():
    """Get a well from a RMS project, reading all logs present."""

    print(roxv)

    if not os.path.isdir(PROJ[roxv]):
        raise RuntimeError('RMS test project is missing for roxar version {}'
                           .format(roxv))

    logger.info('Simple case, reading a well from RMS well folder')

    xwell = xtgeo.well.Well()
    xwell.from_roxar(PROJ[roxv], 'WI_3_RKB2',
                     trajectory='Drilled trajectory',
                     logrun='LOG', lognames='all')

    logger.info('Dataframe\n %s ', xwell.dataframe)

    df = xwell.dataframe
    tsetup.assert_almostequal(df.Poro.mean(), 0.191911, 0.001)
    tsetup.assert_equal(xwell.nrow, 10081, 'NROW of well')
    tsetup.assert_equal(xwell.rkb, -10, 'RKB of well')

    xwell.to_file(join(TMPD, 'roxwell_export.rmswell'))


@tsetup.bigtest
@tsetup.skipunlessroxar
def test_getwell_and_find_ijk_gfb2():
    """Get well from a RMS project, and find IJK from grid."""

    if not os.path.isdir(BPROJ[roxv]):
        pass

    logger.info('GFB case, reading a wells from RMS well folder')

    xwell = xtgeo.well.Well()
    xwell.from_roxar(BPROJ[roxv], '34_10-A-15',
                     trajectory='Drilled trajectory',
                     logrun='data', lognames=['ZONELOG'])

    tsetup.assert_equal(xwell.nrow, 3250, 'NROW of well')
    tsetup.assert_equal(xwell.rkb, 82.20, 'RKB of well')

    # now read a grid
    grd = xtgeo.grid_from_roxar(BPROJ[roxv], 'gfb_sim')

    xwell.make_ijk_from_grid(grd)

    print(xwell.dataframe.head())
    xwell.to_file(join(TMPD, 'gfb2_well_ijk.rmswell'))

    # tsetup.assert_almostequal(x.values.mean(), 1696.255599, 0.001)
