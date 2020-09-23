# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function

import math
from os.path import join

import numpy as np

import test_common.test_xtg as tsetup
import xtgeo

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

TMPD = xtg.tmpdir
TESTPATH = xtg.testpath


# =============================================================================
# Do tests
# =============================================================================

TESTSET1A = "../xtgeo-testdata/surfaces/reek/1/topreek_rota.gri"
TESTSET1B = "../xtgeo-testdata/surfaces/reek/1/basereek_rota.gri"
TESTSETG1 = "../xtgeo-testdata/3dgrids/reek/reek_geo_grid.roff"


def test_create():
    """Create simple Surfaces instance"""

    logger.info("Simple case...")

    top = xtgeo.RegularSurface(TESTSET1A)
    base = xtgeo.RegularSurface(TESTSET1B)
    surfs = xtgeo.Surfaces()
    surfs.surfaces = [top, base]
    surfs.describe()

    assert isinstance(surfs.surfaces[0], xtgeo.RegularSurface)
    assert isinstance(surfs, xtgeo.Surfaces)


def test_create_init_objectlist():
    """Create simple Surfaces instance, initiate with a list of objects"""

    top = xtgeo.RegularSurface(TESTSET1A)
    base = xtgeo.RegularSurface(TESTSET1B)
    surfs = xtgeo.Surfaces([top, base])

    assert isinstance(surfs.surfaces[0], xtgeo.RegularSurface)
    assert isinstance(surfs, xtgeo.Surfaces)


def test_create_init_filelist():
    """Create simple Surfaces instance, initiate with a list of files"""

    flist = [TESTSET1A, TESTSET1B]
    surfs = xtgeo.Surfaces(flist)

    assert isinstance(surfs.surfaces[0], xtgeo.RegularSurface)
    assert isinstance(surfs, xtgeo.Surfaces)


def test_create_init_mixlist():
    """Create simple Surfaces instance, initiate with a list of files"""

    top = xtgeo.RegularSurface(TESTSET1A)
    flist = [top, TESTSET1B]
    surfs = xtgeo.Surfaces(flist)

    assert isinstance(surfs.surfaces[0], xtgeo.RegularSurface)
    assert isinstance(surfs, xtgeo.Surfaces)


def test_statistics():
    """Find the mean etc measures of the surfaces"""

    flist = [TESTSET1A, TESTSET1B]
    surfs = xtgeo.Surfaces(flist)
    res = surfs.statistics()
    res["mean"].to_file(join(TMPD, "surf_mean.gri"))
    res["std"].to_file(join(TMPD, "surf_std.gri"))

    tsetup.assert_almostequal(res["mean"].values.mean(), 1720.5029, 0.0001)
    tsetup.assert_almostequal(res["std"].values.min(), 3.7039, 0.0001)


def test_more_statistics():
    """Find the mean etc measures of the surfaces"""

    base = xtgeo.RegularSurface(TESTSET1A)
    base.values *= 0.0
    bmean = base.values.mean()
    surfs = []
    surfs.append(base)

    # this will get 101 constant maps ranging from 0 til 100
    for inum in range(1, 101):
        tmp = base.copy()
        tmp.values += float(inum)
        surfs.append(tmp)

    so = xtgeo.Surfaces(surfs)
    res = so.statistics()

    # theoretical stdev:
    sum2 = 0.0
    for inum in range(0, 101):
        sum2 += (float(inum) - 50.0) ** 2
    stdev = math.sqrt(sum2 / 100.0)  # total 101 samples, use N-1

    tsetup.assert_almostequal(res["mean"].values.mean(), bmean + 50.0, 0.0001)
    tsetup.assert_almostequal(res["std"].values.mean(), stdev, 0.0001)


def test_surfaces_apply():
    base = xtgeo.RegularSurface(TESTSET1A)
    base.describe()
    base.values *= 0.0
    bmean = base.values.mean()
    surfs = [base]
    for inum in range(1, 101):
        tmp = base.copy()
        tmp.values += float(inum)
        surfs.append(tmp)

    so = xtgeo.Surfaces(surfs)
    res = so.apply(np.nanmean)

    tsetup.assert_almostequal(res.values.mean(), bmean + 50.0, 0.0001)

    res = so.apply(np.nanpercentile, 10, axis=0, interpolation="nearest")
    tsetup.assert_almostequal(res.values.mean(), bmean + 10.0, 0.0001)


def test_get_surfaces_from_3dgrid():
    """Create surfaces from a 3D grid"""

    mygrid = xtgeo.Grid(TESTSETG1)
    surfs = xtgeo.Surfaces()
    surfs.from_grid3d(mygrid, rfactor=2)
    surfs.describe()

    tsetup.assert_almostequal(surfs.surfaces[-1].values.mean(), 1742.28, 0.04)
    tsetup.assert_almostequal(surfs.surfaces[-1].values.min(), 1589.58, 0.04)
    tsetup.assert_almostequal(surfs.surfaces[-1].values.max(), 1977.29, 0.04)
    tsetup.assert_almostequal(surfs.surfaces[0].values.mean(), 1697.02, 0.04)

    for srf in surfs.surfaces:
        srf.to_file(join(TMPD, srf.name + ".gri"))
