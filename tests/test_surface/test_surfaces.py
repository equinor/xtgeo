"""Test surfaces."""
import math
from os.path import join

import numpy as np
import pytest

import xtgeo

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.basiclogger(__name__)

if not xtg.testsetup():
    raise SystemExit

TPATH = xtg.testpathobj


# =============================================================================
# Do tests
# =============================================================================

TESTSET1A = TPATH / "surfaces/reek/1/topreek_rota.gri"
TESTSET1B = TPATH / "surfaces/reek/1/basereek_rota.gri"
TESTSETG1 = TPATH / "3dgrids/reek/reek_geo_grid.roff"


def test_create():
    """Create simple Surfaces instance."""
    logger.info("Simple case...")

    top = xtgeo.surface_from_file(TESTSET1A)
    base = xtgeo.surface_from_file(TESTSET1B)
    surfs = xtgeo.Surfaces()
    surfs.surfaces = [top, base]
    surfs.describe()

    assert isinstance(surfs.surfaces[0], xtgeo.RegularSurface)
    assert isinstance(surfs, xtgeo.Surfaces)


def test_create_init_objectlist():
    """Create simple Surfaces instance, initiate with a list of objects."""
    top = xtgeo.surface_from_file(TESTSET1A)
    base = xtgeo.surface_from_file(TESTSET1B)
    surfs = xtgeo.Surfaces([top, base])

    assert isinstance(surfs.surfaces[0], xtgeo.RegularSurface)
    assert isinstance(surfs, xtgeo.Surfaces)


def test_create_init_filelist():
    """Create simple Surfaces instance, initiate with a list of files."""
    flist = [TESTSET1A, TESTSET1B]
    surfs = xtgeo.Surfaces(flist)

    assert isinstance(surfs.surfaces[0], xtgeo.RegularSurface)
    assert isinstance(surfs, xtgeo.Surfaces)


def test_create_init_mixlist():
    """Create simple Surfaces instance, initiate with a list of files."""
    top = xtgeo.surface_from_file(TESTSET1A)
    flist = [top, TESTSET1B]
    surfs = xtgeo.Surfaces(flist)

    assert isinstance(surfs.surfaces[0], xtgeo.RegularSurface)
    assert isinstance(surfs, xtgeo.Surfaces)


def test_statistics(tmpdir):
    """Find the mean etc measures of the surfaces."""
    flist = [TESTSET1A, TESTSET1B]
    surfs = xtgeo.Surfaces(flist)
    res = surfs.statistics()
    res["mean"].to_file(join(tmpdir, "surf_mean.gri"))
    res["std"].to_file(join(tmpdir, "surf_std.gri"))

    assert res["mean"].values.mean() == pytest.approx(1720.5029, abs=0.0001)
    assert res["std"].values.min() == pytest.approx(3.7039, abs=0.0001)


@pytest.fixture
def constant_map_surfaces():
    base = xtgeo.RegularSurface(ncol=10, nrow=15, xinc=1.0, yinc=1.0, values=0.0)
    surfs = [base]
    # this will get 101 constant maps ranging from 0 til 100
    for inum in range(1, 101):
        tmp = base.copy()
        tmp.values += float(inum)
        surfs.append(tmp)

    return xtgeo.Surfaces(surfs)


def test_more_statistics(constant_map_surfaces):
    res = constant_map_surfaces.statistics()
    # theoretical stdev:
    sum2 = 0.0
    for inum in range(0, 101):
        sum2 += (float(inum) - 50.0) ** 2
    stdev = math.sqrt(sum2 / 100.0)  # total 101 samples, use N-1

    assert res["mean"].values.mean() == pytest.approx(50.0, abs=0.0001)
    assert res["std"].values.mean() == pytest.approx(stdev, abs=0.0001)


@pytest.mark.filterwarnings("ignore:Default values*")
def test_default_surface_statistics(default_surface):
    small = xtgeo.RegularSurface(**default_surface)
    so2 = xtgeo.Surfaces()

    for _ in range(10):
        tmp = small.copy()
        tmp.values += 8.76543
        so2.append([tmp])

    res2 = so2.statistics(percentiles=[10, 50])
    assert res2["p10"].values.mean() == pytest.approx(16.408287142, 0.001)


def test_surfaces_apply(constant_map_surfaces):
    assert constant_map_surfaces.apply(np.nanmean).values.mean() == pytest.approx(
        50.0, abs=0.0001
    )


def test_get_surfaces_from_3dgrid(tmpdir):
    """Create surfaces from a 3D grid."""
    mygrid = xtgeo.grid_from_file(TESTSETG1)
    surfs = xtgeo.surface.surfaces.surfaces_from_grid(mygrid, rfactor=2)
    surfs.describe()

    assert surfs.surfaces[-1].values.mean() == pytest.approx(1742.28, abs=0.04)
    assert surfs.surfaces[-1].values.min() == pytest.approx(1589.58, abs=0.04)
    assert surfs.surfaces[-1].values.max() == pytest.approx(1977.29, abs=0.04)
    assert surfs.surfaces[0].values.mean() == pytest.approx(1697.02, abs=0.04)

    for srf in surfs.surfaces:
        srf.to_file(join(tmpdir, srf.name + ".gri"))
