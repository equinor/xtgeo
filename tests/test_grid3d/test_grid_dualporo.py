# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function

import os

import xtgeo
from xtgeo.common import XTGeoDialog
import test_common.test_xtg as tsetup

xtg = XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

TMPDIR = xtg.tmpdir
TESTPATH = xtg.testpath

DUALFILE1 = "../xtgeo-testdata/3dgrids/etc/TEST_DP"
DUALFILE2 = "../xtgeo-testdata/3dgrids/etc/TEST_DPDK"  # both dual poro and dual perm

# =============================================================================
# Do tests
# =============================================================================


def test_import_dualporo_grid():
    """Test grid with flag for dual porosity setup"""

    grd = xtgeo.grid_from_file(DUALFILE1 + ".EGRID")

    assert grd.dualporo is True
    assert grd.dualperm is False
    assert grd.dimensions == (5, 3, 1)

    poro = xtgeo.gridproperty_from_file(DUALFILE1 + ".INIT", grid=grd, name="PORO")

    tsetup.assert_almostequal(poro.values[0, 0, 0], 0.1, 0.001)
    tsetup.assert_almostequal(poro.values[1, 1, 0], 0.16, 0.001)
    tsetup.assert_almostequal(poro.values[4, 2, 0], 0.24, 0.001)
    assert poro.name == "POROM"
    poro.describe()

    poro = xtgeo.gridproperty_from_file(
        DUALFILE1 + ".INIT", grid=grd, name="PORO", fracture=True
    )

    tsetup.assert_almostequal(poro.values[0, 0, 0], 0.25, 0.001)
    tsetup.assert_almostequal(poro.values[4, 2, 0], 0.39, 0.001)
    assert poro.name == "POROF"
    poro.describe()

    swat = xtgeo.gridproperty_from_file(
        DUALFILE1 + ".UNRST", grid=grd, name="SWAT", date=20170121, fracture=False
    )
    swat.describe()
    tsetup.assert_almostequal(swat.values[0, 0, 0], 0.60924, 0.001)

    swat = xtgeo.gridproperty_from_file(
        DUALFILE1 + ".UNRST", grid=grd, name="SWAT", date=20170121, fracture=True
    )
    swat.describe()
    tsetup.assert_almostequal(swat.values[0, 0, 0], 0.989687, 0.001)
    swat.to_file("TMP/swat.roff")


def test_import_dualperm_grid():
    """Test grid with flag for dual perm setup (will also mean dual poro also)"""

    grd = xtgeo.grid_from_file(DUALFILE2 + ".EGRID")

    assert grd.dualporo is True
    assert grd.dualperm is True
    assert grd.dimensions == (5, 3, 1)
    grd.to_file(os.path.join(TMPDIR, "dual2.roff"))

    poro = xtgeo.gridproperty_from_file(DUALFILE2 + ".INIT", grid=grd, name="PORO")

    tsetup.assert_almostequal(poro.values[0, 0, 0], 0.1, 0.001)
    tsetup.assert_almostequal(poro.values[1, 1, 0], 0.16, 0.001)
    tsetup.assert_almostequal(poro.values[4, 2, 0], 0.24, 0.001)
    assert poro.name == "POROM"
    poro.describe()

    poro = xtgeo.gridproperty_from_file(
        DUALFILE2 + ".INIT", grid=grd, name="PORO", fracture=True
    )

    tsetup.assert_almostequal(poro.values[0, 0, 0], 0.25, 0.001)
    tsetup.assert_almostequal(poro.values[3, 0, 0], 0.0, 0.001)
    tsetup.assert_almostequal(poro.values[4, 2, 0], 0.39, 0.001)
    assert poro.name == "POROF"
    poro.describe()

    perm = xtgeo.gridproperty_from_file(DUALFILE2 + ".INIT", grid=grd, name="PERMX")

    tsetup.assert_almostequal(perm.values[0, 0, 0], 100.0, 0.001)
    tsetup.assert_almostequal(perm.values[3, 0, 0], 100.0, 0.001)
    tsetup.assert_almostequal(perm.values[0, 1, 0], 0.0, 0.001)
    tsetup.assert_almostequal(perm.values[4, 2, 0], 100, 0.001)
    assert perm.name == "PERMXM"
    perm.to_file(os.path.join(TMPDIR, "dual2_permxm.roff"))

    perm = xtgeo.gridproperty_from_file(
        DUALFILE2 + ".INIT", grid=grd, name="PERMX", fracture=True
    )

    tsetup.assert_almostequal(perm.values[0, 0, 0], 100.0, 0.001)
    tsetup.assert_almostequal(perm.values[3, 0, 0], 0.0, 0.001)
    tsetup.assert_almostequal(perm.values[0, 1, 0], 100.0, 0.001)
    tsetup.assert_almostequal(perm.values[4, 2, 0], 100, 0.001)
    assert perm.name == "PERMXF"
    perm.to_file(os.path.join(TMPDIR, "dual2_permxf.roff"))

    swat = xtgeo.gridproperty_from_file(
        DUALFILE2 + ".UNRST", grid=grd, name="SWAT", date=20170121, fracture=False
    )
    swat.describe()
    tsetup.assert_almostequal(swat.values[3, 0, 0], 0.55475, 0.001)

    swat = xtgeo.gridproperty_from_file(
        DUALFILE2 + ".UNRST", grid=grd, name="SWAT", date=20170121, fracture=True
    )
    swat.describe()
    tsetup.assert_almostequal(swat.values[3, 0, 0], 0.0, 0.001)
    swat.to_file("TMP/swat.roff")
