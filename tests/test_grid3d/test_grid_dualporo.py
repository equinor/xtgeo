# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function

import xtgeo
from xtgeo.common import XTGeoDialog
import test_common.test_xtg as tsetup

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

TMPDIR = xtg.tmpdir
TESTPATH = xtg.testpath

DUALFILE = "../xtgeo-testdata/3dgrids/etc/TEST_DP"

# =============================================================================
# Do tests
# =============================================================================


def test_import_dualporo_grid():
    """Test grid with flag for dual porosity setup"""

    grd = xtgeo.grid_from_file(DUALFILE + ".EGRID")

    assert grd.dualporo is True
    assert grd.dimensions == (5, 3, 1)

    poro = xtgeo.gridproperty_from_file(DUALFILE + ".INIT", grid=grd, name="PORO")

    tsetup.assert_almostequal(poro.values[0, 0, 0], 0.1, 0.001)
    tsetup.assert_almostequal(poro.values[1, 1, 0], 0.16, 0.001)
    tsetup.assert_almostequal(poro.values[4, 2, 0], 0.24, 0.001)
    assert poro.name == "POROM"
    poro.describe()

    poro = xtgeo.gridproperty_from_file(
        DUALFILE + ".INIT", grid=grd, name="PORO", fracture=True
    )

    tsetup.assert_almostequal(poro.values[0, 0, 0], 0.25, 0.001)
    tsetup.assert_almostequal(poro.values[4, 2, 0], 0.39, 0.001)
    assert poro.name == "POROF"
    poro.describe()

    swat = xtgeo.gridproperty_from_file(
        DUALFILE + ".UNRST", grid=grd, name="SWAT", date=20170121, fracture=False
    )
    swat.describe()
    tsetup.assert_almostequal(swat.values[0, 0, 0], 0.60924, 0.001)

    swat = xtgeo.gridproperty_from_file(
        DUALFILE + ".UNRST", grid=grd, name="SWAT", date=20170121, fracture=True
    )
    swat.describe()
    tsetup.assert_almostequal(swat.values[0, 0, 0], 0.989687, 0.001)
    swat.to_file("TMP/swat.roff")
