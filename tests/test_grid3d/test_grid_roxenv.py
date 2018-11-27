# coding: utf-8
import pytest
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
project = '/private/jriv/work/testing/reek/reek_grid.rms10.1.1'


@tsetup.skipunlessroxar
def test_rox_get_grid_dimensions_only():
    """Get a grid dimens only from a RMS project."""

    logger.info('Project is {}'.format(project))

    grd = xtgeo.grid3d.Grid()
    grd.from_roxar(project, 'REEK', dimensions_only=True)

    assert grd.dimensions == (40, 64, 14)


@tsetup.skipunlessroxar
def test_rox_get_gridproperty():
    """Get a grid property from a RMS project."""

    logger.info('Project is {}'.format(project))

    grd = xtgeo.grid3d.Grid()
    grd.from_roxar(project, 'REEK')

    grd.to_file('TMP/roxgrid.roff', fformat='roff')

    # tsetup.assert_almostequal(poro.values.mean(), 0.1677, 0.001)
    # tsetup.assert_almostequal(poro.values[1], 0.14942, 0.0001)

    # logger.info('Roxar property id: {}'.format(poro._roxprop))
