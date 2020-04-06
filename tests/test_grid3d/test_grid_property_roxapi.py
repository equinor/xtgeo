# coding: utf-8
"""Testing: test_grid_property_roxenv"""

from __future__ import division, absolute_import
from __future__ import print_function
import os

import xtgeo

import test_common.test_xtg as tsetup

# pylint: disable=logging-format-interpolation
# pylint: disable=invalid-name

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

TESTPATH = "../xtgeo-testdata-equinor/data/rmsprojects"

PROJ = dict()
PROJ["1.1"] = os.path.join(TESTPATH, "reek.rms10.1.1")
PROJ["1.1.1"] = os.path.join(TESTPATH, "reek.rms10.1.3")
PROJ["1.2.1"] = os.path.join(TESTPATH, "reek.rms11.0.1")
PROJ["1.3"] = os.path.join(TESTPATH, "reek.rms11.1.0")


@tsetup.equinor
@tsetup.skipunlessroxar
def test_rox_get_gridproperty():
    """Get a grid property from a RMS project."""
    from roxar import __version__ as ver

    logger.info("Project is {}".format(PROJ[ver]))

    poro = xtgeo.grid3d.GridProperty()
    poro.from_roxar(PROJ[ver], "Reek_sim", "PORO")

    tsetup.assert_almostequal(poro.values.mean(), 0.1588, 0.001)
    assert poro.dimensions == (40, 64, 14)
    tsetup.assert_almostequal(poro.values[1, 0, 0], 0.113876, 0.0001)
    print(poro.values[1, 0, 0])


@tsetup.equinor
@tsetup.skipunlessroxar
def test_rox_get_modify_set_gridproperty():
    """Get and set a grid property from a RMS project."""
    from roxar import __version__ as ver

    poro = xtgeo.grid3d.GridProperty()
    poro.from_roxar(PROJ[ver], "Reek_sim", "PORO")

    adder = 0.9
    poro.values = poro.values + adder

    poro.to_roxar(PROJ[ver], "Reek_sim", "PORO_NEW", saveproject=True)

    poro.from_roxar(PROJ[ver], "Reek_sim", "PORO_NEW")
    tsetup.assert_almostequal(poro.values[1, 0, 0], 0.113876 + adder, 0.0001)
