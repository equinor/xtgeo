# -*- coding: utf-8 -*-

import xtgeo

import tests.test_common.test_xtg as tsetup

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

TPATH = xtg.testpathobj


# =============================================================================
# Do tests
# =============================================================================
RPROJECT = dict()
RPROJECT["1.1"] = "../xtgeo-testdata-equinor/data/rmsprojects/reek.rms10.1.1"
RPROJECT["1.1.1"] = "../xtgeo-testdata-equinor/data/rmsprojects/reek.rms10.1.3"
RPROJECT["1.2.1"] = "../xtgeo-testdata-equinor/data/rmsprojects/reek.rms11.0.1"
RPROJECT["1.3"] = "../xtgeo-testdata-equinor/data/rmsprojects/reek.rms11.1.0"

BPROJECT = dict()
BPROJECT["1.3"] = "../xtgeo-testdata-equinor/data/rmsprojects/bri.rms11.1.0"


@tsetup.equinor
@tsetup.skipunlessroxar
def test_rox_get_grid_dimensions_only():
    """Get a grid dimens only from a RMS project."""
    from roxar import __version__ as ver

    logger.info("Project is {}".format(RPROJECT[ver]))

    grd = xtgeo.grid3d.Grid()
    grd.from_roxar(RPROJECT[ver], "Reek_sim", dimensions_only=True)

    assert grd.dimensions == (40, 64, 14)


@tsetup.equinor
@tsetup.skipunlessroxar
def test_rox_get_grid_import_reek():
    """Get a grid from a RMS project and convert to XTGeo."""
    from roxar import __version__ as ver

    logger.info("Project is {}".format(RPROJECT[ver]))

    grd = xtgeo.grid3d.Grid()

    proj = RPROJECT[ver]
    grd.from_roxar(proj, "Reek_sim", dimensions_only=False, info=True)

    assert grd.dimensions == (40, 64, 14)

    dzprop = grd.get_dz()
    tsetup.assert_almostequal(dzprop.values.mean(), 3.2951, 0.0001)

    # subgrids
    assert grd.subgrids["Below_Mid_reek"] == [6, 7, 8, 9, 10]


@tsetup.equinor
@tsetup.skipunlessroxar
def test_rox_get_grid_import_bri():
    """Get a simple grid from a RMS project and convert to XTGeo."""
    from roxar import __version__ as ver

    if ver not in BPROJECT.keys():
        return

    logger.info("Project is {}".format(BPROJECT[ver]))

    grd = xtgeo.grid3d.Grid()
    grd.from_roxar(BPROJECT[ver], "b_noactnum", dimensions_only=False, info=True)


@tsetup.equinor
@tsetup.skipunlessroxar
def test_rox_get_grid_bri_and_store():
    """Get a simple grid from a RMS project and store again"""
    from roxar import __version__ as ver

    if ver not in BPROJECT.keys():
        return

    logger.info("Project is {}".format(BPROJECT[ver]))

    grd = xtgeo.grid3d.Grid()
    grd.from_roxar(BPROJECT[ver], "b_noactnum", dimensions_only=False, info=False)

    grd.to_roxar(BPROJECT[ver], "b_noactnum", info=True)
