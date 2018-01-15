# coding: utf-8

import xtgeo
import tests.test_setup as tsetup

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
def test_rox_get_gridproperty():
    """Get a grid property from a RMS project."""

    logger.info('Project is {}'.format(project))

    poro = xtgeo.grid3d.GridProperty()
    poro.from_roxar(project, 'REEK', 'PORO')

    tsetup.assert_almostequal(poro.values.mean(), 0.1677, 0.001)
    tsetup.assert_almostequal(poro.values[1], 0.14942, 0.0001)

    # logger.info('Roxar property id: {}'.format(poro._roxprop))


@tsetup.skipunlessroxar
def test_rox_get_modify_set_gridproperty():
    """Get and set a grid property from a RMS project."""

    logger.info('Project is {}'.format(project))

    poro = xtgeo.grid3d.GridProperty()
    poro.from_roxar(project, 'REEK', 'PORO')

    adder = 0.9
    poro.values = poro.values + adder

    poro.to_roxar(project, 'REEK', 'PORO_NEW', saveproject=True)

    poro.from_roxar(project, 'REEK', 'PORO_NEW')
    tsetup.assert_almostequal(poro.values[1], 0.14942 + adder, 0.0001)
