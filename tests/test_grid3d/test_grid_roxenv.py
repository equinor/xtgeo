# -*- coding: utf-8 -*-

import xtgeo

import test_common.test_xtg as tsetup

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

td = xtg.tmpdir
testpath = xtg.testpath


# =============================================================================
# Do tests
# =============================================================================
rproject = dict()
rproject['1.1'] = '../xtgeo-testdata-equinor/data/rmsprojects/reek.rms10.1.1'
rproject['1.2.1'] = '../xtgeo-testdata-equinor/data/rmsprojects/reek.rms11.0.1'
rproject['1.3'] = '../xtgeo-testdata-equinor/data/rmsprojects/reek.rms11.1.0'

bproject = dict()
bproject['1.3'] = '../xtgeo-testdata-equinor/data/rmsprojects/bri.rms11.1.0'


@tsetup.equinor
@tsetup.skipunlessroxar
def test_rox_get_grid_dimensions_only():
    """Get a grid dimens only from a RMS project."""
    from roxar import __version__ as ver

    logger.info('Project is {}'.format(rproject[ver]))

    grd = xtgeo.grid3d.Grid()
    grd.from_roxar(rproject[ver], 'Reek_sim', dimensions_only=True)

    assert grd.dimensions == (40, 64, 14)


@tsetup.equinor
@tsetup.skipunlessroxar
def test_rox_get_grid_import_reek():
    """Get a grid from a RMS project and convert to XTGeo."""
    from roxar import __version__ as ver

    logger.info('Project is {}'.format(rproject[ver]))

    grd = xtgeo.grid3d.Grid()
    grd.from_roxar(rproject[ver], 'Reek_sim', dimensions_only=False, info=True)

    assert grd.dimensions == (40, 64, 14)

    dzprop = grd.get_dz()
    tsetup.assert_almostequal(dzprop.values.mean(), 3.2951, 0.0001)


@tsetup.equinor
@tsetup.skipunlessroxar
def test_rox_get_grid_import_bri():
    """Get a simple grid from a RMS project and convert to XTGeo."""
    from roxar import __version__ as ver

    logger.info('Project is {}'.format(bproject[ver]))

    grd = xtgeo.grid3d.Grid()
    grd.from_roxar(bproject[ver], 'b_noactnum', dimensions_only=False,
                   info=True)
