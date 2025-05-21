"""Test surfaces."""

import math
import pathlib

import numpy as np
import pytest

import xtgeo

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.basiclogger(__name__)

TESTSET1A = pathlib.Path("surfaces/reek/1/topreek_rota.gri")
TESTSET1B = pathlib.Path("surfaces/reek/1/basereek_rota.gri")
TESTSETG1 = pathlib.Path("3dgrids/reek/reek_geo_grid.roff")


def test_create(testdata_path):
    """Create simple Surfaces instance."""
    logger.info("Simple case...")

    top = xtgeo.surface_from_file(testdata_path / TESTSET1A)
    base = xtgeo.surface_from_file(testdata_path / TESTSET1B)
    surfs = xtgeo.Surfaces()
    surfs.surfaces = [top, base]
    surfs.describe()

    assert isinstance(surfs.surfaces[0], xtgeo.RegularSurface)
    assert isinstance(surfs, xtgeo.Surfaces)


def test_create_init_objectlist(testdata_path):
    """Create simple Surfaces instance, initiate with a list of objects."""
    top = xtgeo.surface_from_file(testdata_path / TESTSET1A)
    base = xtgeo.surface_from_file(testdata_path / TESTSET1B)
    surfs = xtgeo.Surfaces([top, base])

    assert isinstance(surfs.surfaces[0], xtgeo.RegularSurface)
    assert isinstance(surfs, xtgeo.Surfaces)


def test_create_init_filelist(testdata_path):
    """Create simple Surfaces instance, initiate with a list of files."""
    flist = [testdata_path / TESTSET1A, testdata_path / TESTSET1B]
    surfs = xtgeo.Surfaces(flist)

    assert isinstance(surfs.surfaces[0], xtgeo.RegularSurface)
    assert isinstance(surfs, xtgeo.Surfaces)


def test_create_init_mixlist(testdata_path):
    """Create simple Surfaces instance, initiate with a list of files."""
    top = xtgeo.surface_from_file(testdata_path / TESTSET1A)
    flist = [top, testdata_path / TESTSET1B]
    surfs = xtgeo.Surfaces(flist)

    assert isinstance(surfs.surfaces[0], xtgeo.RegularSurface)
    assert isinstance(surfs, xtgeo.Surfaces)


def test_statistics(tmp_path, testdata_path):
    """Find the mean etc measures of the surfaces."""
    flist = [testdata_path / TESTSET1A, testdata_path / TESTSET1B]
    surfs = xtgeo.Surfaces(flist)
    res = surfs.statistics()
    res["mean"].to_file(tmp_path / "surf_mean.gri")
    res["std"].to_file(tmp_path / "surf_std.gri")

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
    for inum in range(101):
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


def test_get_surfaces_from_3dgrid(tmp_path, testdata_path):
    """Create surfaces from a 3D grid."""
    mygrid = xtgeo.grid_from_file(testdata_path / TESTSETG1)
    surfs = xtgeo.surface.surfaces.surfaces_from_grid(mygrid, rfactor=2)
    surfs.describe()

    for srf in surfs.surfaces:
        srf.to_file(tmp_path / pathlib.Path(f"{srf.name}.gri"))

    assert surfs.surfaces[-1].values.mean() == pytest.approx(1742.2699, abs=0.04)
    assert surfs.surfaces[-1].values.min() == pytest.approx(1590.5657, abs=0.04)
    assert surfs.surfaces[-1].values.max() == pytest.approx(1977.2443, abs=0.04)
    assert surfs.surfaces[0].values.mean() == pytest.approx(1697.0290, abs=0.04)


def test_is_depth_consistent():
    """Test surface depth consistency checking."""
    # Create three surfaces with known geometry
    surf1 = xtgeo.RegularSurface(ncol=5, nrow=3, xinc=1.0, yinc=1.0)
    surf2 = xtgeo.RegularSurface(ncol=5, nrow=3, xinc=1.0, yinc=1.0)
    surf3 = xtgeo.RegularSurface(ncol=5, nrow=3, xinc=1.0, yinc=1.0)

    # Set values for each surface (increasing depth)
    surf1.values = np.full((3, 5), 100.0)
    surf2.values = np.full((3, 5), 200.0)
    surf3.values = np.full((3, 5), 300.0)

    # Test consistent surfaces
    surfs = xtgeo.Surfaces([surf1, surf2, surf3])
    assert surfs.is_depth_consistent() is True

    # Test inconsistent surfaces by making one point cross
    surf2.values[1, 1] = 50.0  # Make surface 2 cross surface 1
    assert surfs.is_depth_consistent() is False

    # Test topology mismatch
    surf4 = xtgeo.RegularSurface(ncol=6, nrow=3, xori=0.0, yori=0.0, xinc=1.0, yinc=1.0)
    with pytest.raises(ValueError, match=".*topology.*"):
        xtgeo.Surfaces([surf1, surf4]).is_depth_consistent()


def test_make_depth_consistent():
    """Test making surfaces depth consistent."""
    # Create test surfaces
    surf1 = xtgeo.RegularSurface(ncol=3, nrow=3, xinc=100.0, yinc=100.0)
    surf2 = xtgeo.RegularSurface(ncol=3, nrow=3, xinc=100.0, yinc=100.0)
    surf3 = xtgeo.RegularSurface(ncol=3, nrow=3, xinc=100.0, yinc=100.0)

    # Set initial values with some crossings
    surf1.values = np.array(
        [[100.0, 100.0, 100.0], [100.0, 100.0, 100.0], [100.0, 100.0, 100.0]]
    )

    surf2.values = np.array(
        [
            [200.0, 90.0, 200.0],  # 90 crosses surf1
            [200.0, 200.0, 200.0],
            [200.0, 200.0, 200.0],
        ]
    )

    surf3.values = np.array(
        [
            [300.0, 300.0, 300.0],
            [150.0, 300.0, 300.0],  # 150 crosses surf2
            [300.0, 300.0, 300.0],
        ]
    )

    # Test inplace=False
    surfs = xtgeo.Surfaces([surf1, surf2, surf3])
    new_surfs = surfs.make_depth_consistent(inplace=False)
    assert new_surfs is not surfs  # Should be different object
    assert new_surfs.is_depth_consistent()

    # Original should be unchanged
    assert surfs.surfaces[1].values[0, 1] == 90.0
    assert surfs.surfaces[2].values[1, 0] == 150.0

    # Test inplace=True
    surfs = xtgeo.Surfaces([surf1, surf2, surf3])
    result = surfs.make_depth_consistent(inplace=True)
    assert result is None  # Should return None for inplace=True

    # Check values were corrected
    assert surfs.surfaces[1].values[0, 1] == 100.0  # surf2 adjusted to surf1
    assert surfs.surfaces[2].values[1, 0] == 200.0  # surf3 adjusted to surf2
    assert surfs.is_depth_consistent()

    # Test topology mismatch
    surf4 = xtgeo.RegularSurface(ncol=4, nrow=3, xinc=100.0, yinc=100.0)
    surfs_bad = xtgeo.Surfaces([surf1, surf4])
    with pytest.raises(ValueError, match=".*topology.*"):
        surfs_bad.make_depth_consistent()
