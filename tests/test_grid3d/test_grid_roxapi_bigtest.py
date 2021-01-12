# -*- coding: utf-8 -*-

import xtgeo

import tests.test_common.test_xtg as tsetup

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

td = xtg.tmpdir
testpath = xtg.testpathobj

BPROJECT = dict()
BPROJECT["1.1.1"] = "../xtgeo-testdata-equinor/data/rmsprojects/gfb.rms10.1.3"
BPROJECT["1.3"] = "../xtgeo-testdata-equinor/data/rmsprojects/gfb.rms11.1.0"


@tsetup.equinor
@tsetup.bigtest
@tsetup.skipunlessroxar
def test_rox_get_grid_dimensions_only():
    """Get a grid dimens only from a RMS project."""
    from roxar import __version__ as ver

    logger.info("Project is {}".format(BPROJECT[ver]))

    grd = xtgeo.grid3d.Grid()
    grd.from_roxar(BPROJECT[ver], "gfb_gg", dimensions_only=True)

    assert grd.dimensions == (198, 240, 394)


@tsetup.equinor
@tsetup.bigtest
@tsetup.skipunlessroxar
def test_rox_get_grid_import_gfb_geo():
    """Get a grid from a large RMS grid and convert to XTGeo."""
    from roxar import __version__ as ver

    logger.info("Project is {}".format(BPROJECT[ver]))

    grd = xtgeo.grid3d.Grid()
    grd.from_roxar(BPROJECT[ver], "gfb_gg", dimensions_only=False, info=True)

    assert grd.dimensions == (198, 240, 394)

    dzprop = grd.get_dz()
    tsetup.assert_almostequal(dzprop.values.mean(), 0.614318, 0.0001)
