# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function

import os
import numpy as np

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
DUALFILE2 = "../xtgeo-testdata/3dgrids/etc/TEST_DPDK"  # dual poro + dual perm oil/water
DUALFILE3 = "../xtgeo-testdata/3dgrids/etc/TEST2_DPDK_WG"  # aa but gas/water

# =============================================================================
# Do tests
# =============================================================================


def test_import_dualporo_grid():
    """Test grid with flag for dual porosity setup, oil water"""

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
    """Test grid with flag for dual perm setup (hence dual poro also) water/oil"""

    grd = xtgeo.grid_from_file(DUALFILE2 + ".EGRID")

    assert grd.dualporo is True
    assert grd.dualperm is True
    assert grd.dimensions == (5, 3, 1)
    grd.to_file(os.path.join(TMPDIR, "dual2.roff"))

    poro = xtgeo.gridproperty_from_file(DUALFILE2 + ".INIT", grid=grd, name="PORO")
    print(poro.values)

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
    tsetup.assert_almostequal(swat.values[3, 0, 0], 0.55475, 0.001)

    soil = xtgeo.gridproperty_from_file(
        DUALFILE2 + ".UNRST", grid=grd, name="SOIL", date=20170121, fracture=False
    )
    print(soil.values)
    tsetup.assert_almostequal(soil.values[3, 0, 0], 0.44525, 0.001)
    tsetup.assert_almostequal(soil.values[0, 1, 0], 0.0, 0.001)
    assert np.ma.is_masked(soil.values[1, 2, 0])
    tsetup.assert_almostequal(soil.values[3, 2, 0], 0.0, 0.001)
    tsetup.assert_almostequal(soil.values[4, 2, 0], 0.41271, 0.001)

    swat = xtgeo.gridproperty_from_file(
        DUALFILE2 + ".UNRST", grid=grd, name="SWAT", date=20170121, fracture=True
    )
    swat.describe()
    assert "SWATF" in swat.name

    tsetup.assert_almostequal(swat.values[3, 0, 0], 0.0, 0.001)
    swat.to_file("TMP/swat.roff")


def test_import_dualperm_grid_soil():
    """Test grid with flag for dual perm setup (will also mean dual poro also)"""

    grd = xtgeo.grid_from_file(DUALFILE2 + ".EGRID")
    grd._dualactnum.to_file("TMP/dualact.roff")

    sgas = xtgeo.gridproperty_from_file(
        DUALFILE2 + ".UNRST", grid=grd, name="SGAS", date=20170121, fracture=False
    )
    sgas.describe()
    tsetup.assert_almostequal(sgas.values[3, 0, 0], 0.0, 0.001)
    tsetup.assert_almostequal(sgas.values[0, 1, 0], 0.0, 0.001)

    soil = xtgeo.gridproperty_from_file(
        DUALFILE2 + ".UNRST", grid=grd, name="SOIL", date=20170121, fracture=False
    )
    soil.describe()
    tsetup.assert_almostequal(soil.values[3, 0, 0], 0.44525, 0.001)
    tsetup.assert_almostequal(soil.values[0, 1, 0], 0.0, 0.001)
    tsetup.assert_almostequal(soil.values[3, 2, 0], 0.0, 0.0001)

    # fractures

    sgas = xtgeo.gridproperty_from_file(
        DUALFILE2 + ".UNRST", grid=grd, name="SGAS", date=20170121, fracture=True
    )
    tsetup.assert_almostequal(sgas.values[3, 0, 0], 0.0, 0.001)
    tsetup.assert_almostequal(sgas.values[0, 1, 0], 0.0, 0.0001)

    soil = xtgeo.gridproperty_from_file(
        DUALFILE2 + ".UNRST", grid=grd, name="SOIL", date=20170121, fracture=True
    )
    tsetup.assert_almostequal(soil.values[3, 0, 0], 0.0, 0.001)
    tsetup.assert_almostequal(soil.values[0, 1, 0], 0.011741, 0.0001)
    tsetup.assert_almostequal(soil.values[3, 2, 0], 0.11676, 0.0001)


def test_import_dualperm_grid_sgas():
    """Test grid with flag for dual perm/poro setup gas/water"""

    grd = xtgeo.grid_from_file(DUALFILE3 + ".EGRID")

    sgas = xtgeo.gridproperty_from_file(
        DUALFILE3 + ".UNRST", grid=grd, name="SGAS", date=20170121, fracture=False
    )
    sgas.describe()
    tsetup.assert_almostequal(sgas.values[3, 0, 0], 0.06639, 0.001)
    tsetup.assert_almostequal(sgas.values[0, 1, 0], 0.0, 0.001)
    tsetup.assert_almostequal(sgas.values[4, 2, 0], 0.10696, 0.001)
    assert "SGASM in sgas.name"

    swat = xtgeo.gridproperty_from_file(
        DUALFILE3 + ".UNRST", grid=grd, name="SWAT", date=20170121, fracture=False
    )
    swat.describe()
    tsetup.assert_almostequal(swat.values[3, 0, 0], 0.93361, 0.001)
    tsetup.assert_almostequal(swat.values[0, 1, 0], 0.0, 0.001)
    tsetup.assert_almostequal(swat.values[4, 2, 0], 0.89304, 0.001)
    assert "SWATM in swat.name"

    # shall be not soil actually
    soil = xtgeo.gridproperty_from_file(
        DUALFILE3 + ".UNRST", grid=grd, name="SOIL", date=20170121, fracture=False
    )
    soil.describe()
    tsetup.assert_almostequal(soil.values[3, 0, 0], 0.0, 0.001)
    tsetup.assert_almostequal(soil.values[0, 1, 0], 0.0, 0.001)
    assert "SOILM" in soil.name

    # fractures

    sgas = xtgeo.gridproperty_from_file(
        DUALFILE3 + ".UNRST", grid=grd, name="SGAS", date=20170121, fracture=True
    )
    sgas.describe()
    tsetup.assert_almostequal(sgas.values[3, 0, 0], 0.0, 0.001)
    tsetup.assert_almostequal(sgas.values[0, 1, 0], 0.0018198, 0.001)
    tsetup.assert_almostequal(sgas.values[4, 2, 0], 0.17841, 0.001)
    assert "SGASF" in sgas.name

    swat = xtgeo.gridproperty_from_file(
        DUALFILE3 + ".UNRST", grid=grd, name="SWAT", date=20170121, fracture=True
    )
    swat.describe()
    tsetup.assert_almostequal(swat.values[3, 0, 0], 0.0, 0.001)
    tsetup.assert_almostequal(swat.values[0, 1, 0], 0.99818, 0.001)
    tsetup.assert_almostequal(swat.values[4, 2, 0], 0.82159, 0.001)
    assert "SWATF" in swat.name

    # shall be not soil actually
    soil = xtgeo.gridproperty_from_file(
        DUALFILE3 + ".UNRST", grid=grd, name="SOIL", date=20170121, fracture=True
    )
    soil.describe()
    tsetup.assert_almostequal(soil.values[3, 0, 0], 0.0, 0.001)
    tsetup.assert_almostequal(soil.values[0, 1, 0], 0.0, 0.001)
    assert "SOILF" in soil.name
