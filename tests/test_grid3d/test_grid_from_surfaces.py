import pathlib

import numpy as np
import pytest

import xtgeo
from xtgeo import RegularSurface, Surfaces

SURFACES = [
    pathlib.Path("surfaces/drogon/1/01_topvolantis.gri"),
    pathlib.Path("surfaces/drogon/1/02_toptherys.gri"),
    pathlib.Path("surfaces/drogon/1/03_topvolon.gri"),
    pathlib.Path("surfaces/drogon/1/04_basevolantis.gri"),
]


@pytest.fixture
def load_surfaces(testdata_path):
    surfaces = xtgeo.Surfaces()
    for surf in SURFACES:
        surfaces.append(xtgeo.surface_from_file(testdata_path / surf))

    return surfaces


def test_grid_from_surfaces_basic1():
    """Test creating a grid from basic surfaces."""
    # Create two surfaces with known geometry
    surf1 = RegularSurface(ncol=5, nrow=3, xori=0.0, yori=0.0, xinc=100.0, yinc=100.0)
    surf2 = RegularSurface(ncol=5, nrow=3, xori=0.0, yori=0.0, xinc=100.0, yinc=100.0)

    # Set values for each surface (depth increasing)
    surf1.values = np.full((3, 5), 1000.0)
    surf2.values = np.full((3, 5), 1200.0)

    # Create surfaces collection
    surfs = Surfaces([surf1, surf2])

    # Create grid
    grid = xtgeo.grid_from_surfaces(surfs, tolerance=1e-6)

    # Verify grid dimensions (one less than surfaces for col, row)
    assert grid.ncol == 4
    assert grid.nrow == 2
    assert grid.nlay == 1  # one layer between two surfaces

    # Verify grid properties
    assert grid.nactive == grid.ntotal  # all cells should be active

    # Get cell center depths and verify they are between surfaces
    xyz = grid.get_xyz()
    z_values = xyz[2].values

    assert np.all(z_values >= 1000.0)  # above or equal to top surface
    assert np.all(z_values <= 1200.0)  # below or equal to base surface


def test_grid_from_surfaces_tolerance():
    """Test creating a grid from basic surfaces, adjusting tolerance."""
    # Create two surfaces with known geometry
    surf1 = RegularSurface(ncol=5, nrow=3, xori=0.0, yori=0.0, xinc=100.0, yinc=100.0)
    surf2 = RegularSurface(ncol=5, nrow=3, xori=0.0, yori=0.0, xinc=100.0, yinc=100.0)

    # Set values for each surface (depth increasing)
    surf1.values = np.full((3, 5), 1000.0)
    surf2.values = np.full((3, 5), 1200.0)

    # Create surfaces collection
    surfs = Surfaces([surf1, surf2])

    # Create grid
    grid = xtgeo.grid_from_surfaces(
        surfs, ij_increment=(100.0000001, 100.0000001), tolerance=1e-2
    )
    assert grid.nactive == 8

    grid = xtgeo.grid_from_surfaces(
        surfs, ij_increment=(100.0000001, 100.0000001), tolerance=1e-12
    )
    assert grid.nactive == 3  # less active cells due to precision


def test_grid_from_surfaces_with_parameters():
    """Test creating a grid with custom dimensions and increments."""
    surf1 = RegularSurface(ncol=10, nrow=10, xinc=50.0, yinc=50.0)
    surf2 = RegularSurface(ncol=10, nrow=10, xinc=50.0, yinc=50.0)

    surf1.values = np.full((10, 10), 1000.0)
    surf2.values = np.full((10, 10), 1200.0)

    surfs = Surfaces([surf1, surf2])

    # Create grid with custom parameters
    grid = xtgeo.grid_from_surfaces(
        surfs,
        ij_dimension=(5, 5),  # coarser grid than surfaces
        ij_increment=(100.0, 100.0),  # different increments
        rotation=30.0,  # add rotation
    )

    assert grid.ncol == 5
    assert grid.nrow == 5
    assert grid.nlay == 1


def test_grid_from_surfaces_errors():
    """Test error conditions when creating grid from surfaces."""
    # Create surfaces with different dimensions
    surf1 = RegularSurface(ncol=5, nrow=3, xinc=50.0, yinc=50.0)
    surf2 = RegularSurface(ncol=6, nrow=3, xinc=50.0, yinc=50.0)  # different ncol

    with pytest.raises(ValueError):
        surfs = Surfaces([surf1, surf2])
        xtgeo.grid_from_surfaces(surfs)

    # Create surfaces that cross each other
    surf1 = RegularSurface(ncol=5, nrow=3, xinc=50.0, yinc=50.0)
    surf2 = RegularSurface(ncol=5, nrow=3, xinc=50.0, yinc=50.0)

    surf1.values = np.full((3, 5), 1200.0)  # top surface deeper than base
    surf2.values = np.full((3, 5), 1000.0)

    with pytest.raises(ValueError):
        surfs = Surfaces([surf1, surf2])
        xtgeo.grid_from_surfaces(surfs)


@pytest.mark.parametrize(
    "increment, rotation, tolerance, expected_nactive, expected_avg_dz",
    [
        (None, None, 1e-6, 142461, 13.98784),  # default tolerance is 1e-6
        (None, None, 0.1, 143028, 13.987079),  # ...activate more edge cells
        ((100, 100), 0.0, 1e-6, 12171, 14.684326),
        ((50, 150), 20.0, 1e-6, 25875, 14.196919),
    ],
)
def test_grid_from_surfaces_drogon(
    load_surfaces, increment, rotation, tolerance, expected_nactive, expected_avg_dz
):
    """Testing a real case, assertions are validated in graphical tools like RMS."""

    surfaces = load_surfaces

    grid = xtgeo.grid_from_surfaces(
        surfaces, ij_increment=increment, rotation=rotation, tolerance=tolerance
    )

    assert grid.nactive == expected_nactive

    assert grid.get_dz().values.mean() == pytest.approx(expected_avg_dz)
