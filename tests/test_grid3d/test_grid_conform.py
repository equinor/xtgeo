"""Tests for conform_to_surfaces grid method."""

import numpy as np
import pytest

from xtgeo import RegularSurface
from xtgeo.grid3d.grid import create_box_grid


def _make_flat_surface(ncol, nrow, xinc, yinc, depth, xori=0.0, yori=0.0):
    """Helper to create a flat surface at a given depth."""
    surf = RegularSurface(
        ncol=ncol, nrow=nrow, xinc=xinc, yinc=yinc, xori=xori, yori=yori
    )
    surf.values = np.full((ncol, nrow), depth)
    return surf


def _make_tilted_surface(
    ncol, nrow, xinc, yinc, depth_min, depth_max, xori=0.0, yori=0.0
):
    """Helper to create a surface tilted in the x direction."""
    surf = RegularSurface(
        ncol=ncol, nrow=nrow, xinc=xinc, yinc=yinc, xori=xori, yori=yori
    )
    values = np.zeros((ncol, nrow))
    for i in range(ncol):
        frac = i / max(ncol - 1, 1)
        values[i, :] = depth_min + frac * (depth_max - depth_min)
    surf.values = values
    return surf


def _make_faulted_grid(nlay=4):
    """Create a 3x3 grid with a fault at pillar column i=2 (50m throw)."""
    grid = create_box_grid(
        dimension=(3, 3, nlay),
        origin=(0.0, 0.0, 1000.0),
        increment=(100.0, 100.0, 50.0),
    )
    for j in range(grid.nrow + 1):
        for k in range(grid.nlay + 1):
            grid._zcornsv[2, j, k, 1] += 50.0
            grid._zcornsv[2, j, k, 3] += 50.0
    return grid


# --- Core behaviour ---


def test_single_zone_flat_surfaces():
    """Single zone: ZCORN snapped to surfaces, COORD unchanged."""
    grid = create_box_grid(
        dimension=(4, 3, 5),
        origin=(0.0, 0.0, 1000.0),
        increment=(100.0, 100.0, 10.0),
    )
    original_coord = grid._coordsv.copy()
    surf_top = _make_flat_surface(5, 4, 100.0, 100.0, 900.0)
    surf_bot = _make_flat_surface(5, 4, 100.0, 100.0, 1100.0)

    grid.conform_to_surfaces(surfaces=[surf_top, surf_bot], layers_per_zone=[5])

    dz = (1100.0 - 900.0) / 5.0
    for k in range(6):
        assert np.allclose(grid._zcornsv[:, :, k, :], 900.0 + k * dz, atol=0.01)
    np.testing.assert_array_equal(grid._coordsv, original_coord)


def test_two_zones_proportional_distribution():
    """Two zones with different thicknesses; subgrids set correctly."""
    grid = create_box_grid(
        dimension=(3, 3, 6),
        origin=(0.0, 0.0, 0.0),
        increment=(100.0, 100.0, 10.0),
    )
    surfs = [
        _make_flat_surface(4, 4, 100.0, 100.0, 100.0),
        _make_flat_surface(4, 4, 100.0, 100.0, 400.0),
        _make_flat_surface(4, 4, 100.0, 100.0, 1000.0),
    ]

    grid.conform_to_surfaces(surfaces=surfs, layers_per_zone=[3, 3])

    # Zone 1: 100->400, dz=100; Zone 2: 400->1000, dz=200
    for k in range(3):
        dz = grid._zcornsv[1, 1, k + 1, 0] - grid._zcornsv[1, 1, k, 0]
        assert np.isclose(dz, 100.0, atol=0.01)
    for k in range(3, 6):
        dz = grid._zcornsv[1, 1, k + 1, 0] - grid._zcornsv[1, 1, k, 0]
        assert np.isclose(dz, 200.0, atol=0.01)

    assert grid.get_subgrids() == {"zone_1": 3, "zone_2": 3}


def test_tilted_surface():
    """Zone boundaries vary with x position for tilted surfaces."""
    grid = create_box_grid(
        dimension=(4, 3, 4),
        origin=(0.0, 0.0, 1000.0),
        increment=(100.0, 100.0, 10.0),
    )
    surf_top = _make_tilted_surface(5, 4, 100.0, 100.0, 800.0, 1200.0)
    surf_bot = _make_tilted_surface(5, 4, 100.0, 100.0, 1200.0, 1600.0)

    grid.conform_to_surfaces(surfaces=[surf_top, surf_bot], layers_per_zone=[4])

    assert np.isclose(grid._zcornsv[0, 0, 0, 0], 800.0, atol=1.0)
    assert np.isclose(grid._zcornsv[0, 0, 4, 0], 1200.0, atol=1.0)
    assert np.isclose(grid._zcornsv[4, 0, 0, 0], 1200.0, atol=1.0)
    assert np.isclose(grid._zcornsv[4, 0, 4, 0], 1600.0, atol=1.0)


def test_single_layer():
    """Minimum grid: single layer snapped to two surfaces."""
    grid = create_box_grid(
        dimension=(3, 3, 1),
        origin=(0.0, 0.0, 0.0),
        increment=(100.0, 100.0, 100.0),
    )
    surfs = [
        _make_flat_surface(4, 4, 100.0, 100.0, 500.0),
        _make_flat_surface(4, 4, 100.0, 100.0, 600.0),
    ]

    grid.conform_to_surfaces(surfaces=surfs, layers_per_zone=[1])

    assert np.allclose(grid._zcornsv[:, :, 0, :], 500.0, atol=0.01)
    assert np.allclose(grid._zcornsv[:, :, 1, :], 600.0, atol=0.01)


def test_idempotent():
    """Conforming to surfaces matching original grid is a no-op."""
    grid = create_box_grid(
        dimension=(3, 3, 4),
        origin=(0.0, 0.0, 1000.0),
        increment=(100.0, 100.0, 50.0),
    )
    surfs = [
        _make_flat_surface(4, 4, 100.0, 100.0, 1000.0),
        _make_flat_surface(4, 4, 100.0, 100.0, 1200.0),
    ]
    zcorn_before = grid._zcornsv.copy()

    grid.conform_to_surfaces(surfaces=surfs, layers_per_zone=[4])

    np.testing.assert_allclose(grid._zcornsv, zcorn_before, atol=0.1)


# --- Inclined pillars ---


def test_inclined_pillars():
    """Inclined pillars: COORD preserved, ZCORN updated to surface values."""
    grid = create_box_grid(
        dimension=(3, 3, 4),
        origin=(100.0, 100.0, 1000.0),
        increment=(100.0, 100.0, 50.0),
    )
    grid._coordsv[:, :, 3] += 20.0  # X shift
    grid._coordsv[:, :, 4] += 10.0  # Y shift
    original_coord = grid._coordsv.copy()

    surfs = [
        _make_flat_surface(8, 8, 100.0, 100.0, 900.0),
        _make_flat_surface(8, 8, 100.0, 100.0, 1300.0),
    ]

    grid.conform_to_surfaces(surfaces=surfs, layers_per_zone=[4])

    np.testing.assert_allclose(grid._coordsv, original_coord, atol=1e-10)
    assert np.allclose(grid._zcornsv[:, :, 0, :], 900.0, atol=1.0)
    assert np.allclose(grid._zcornsv[:, :, 4, :], 1300.0, atol=1.0)


# --- Faulted grids ---


def test_faulted_preserves_throw():
    """Fault throw is preserved: corner averages match surfaces, no inversions."""
    grid = _make_faulted_grid()
    surfs = [
        _make_flat_surface(4, 4, 100.0, 100.0, 800.0),
        _make_flat_surface(4, 4, 100.0, 100.0, 1400.0),
    ]

    grid.conform_to_surfaces(surfaces=surfs, layers_per_zone=[4])

    for j in range(grid.nrow + 1):
        assert np.isclose(np.mean(grid._zcornsv[2, j, 0, :]), 800.0, atol=1.0)
        assert np.isclose(np.mean(grid._zcornsv[2, j, 4, :]), 1400.0, atol=1.0)
        for c in range(4):
            for k in range(grid.nlay):
                assert grid._zcornsv[2, j, k + 1, c] >= grid._zcornsv[2, j, k, c] - 0.01


def test_faulted_two_zone_distribution():
    """Proportional layer distribution at faulted pillars with two zones."""
    grid = _make_faulted_grid()
    surfs = [
        _make_flat_surface(4, 4, 100.0, 100.0, 900.0),
        _make_flat_surface(4, 4, 100.0, 100.0, 1100.0),
        _make_flat_surface(4, 4, 100.0, 100.0, 1500.0),
    ]

    grid.conform_to_surfaces(surfaces=surfs, layers_per_zone=[2, 2])

    expected = [900.0, 1000.0, 1100.0, 1300.0, 1500.0]
    for j in range(grid.nrow + 1):
        for k, z_exp in enumerate(expected):
            assert np.isclose(np.mean(grid._zcornsv[2, j, k, :]), z_exp, atol=1.0)


def test_pinched_faulted():
    """Pinched faulted grid: no inversions, zone boundaries match surfaces."""
    grid = _make_faulted_grid(nlay=6)
    # Pinch layers 2-3 on the east side of fault at pillar i=2
    pinch_z = 1100.0 + 50.0  # = 1150
    for j in range(grid.nrow + 1):
        for k in [2, 3, 4]:
            grid._zcornsv[2, j, k, 1] = pinch_z
            grid._zcornsv[2, j, k, 3] = pinch_z

    surfs = [
        _make_flat_surface(4, 4, 100.0, 100.0, 900.0),
        _make_flat_surface(4, 4, 100.0, 100.0, 1200.0),
        _make_flat_surface(4, 4, 100.0, 100.0, 1500.0),
    ]

    grid.conform_to_surfaces(surfaces=surfs, layers_per_zone=[3, 3])

    # No inversions anywhere
    for i in range(grid.ncol + 1):
        for j in range(grid.nrow + 1):
            for c in range(4):
                for k in range(grid.nlay):
                    assert (
                        grid._zcornsv[i, j, k + 1, c]
                        >= grid._zcornsv[i, j, k, c] - 0.01
                    )

    # Zone boundaries match surfaces at non-faulted pillars
    for i in [0, 1, 3]:
        for j in range(grid.nrow + 1):
            assert np.isclose(np.mean(grid._zcornsv[i, j, 0, :]), 900.0, atol=1.0)
            assert np.isclose(np.mean(grid._zcornsv[i, j, 3, :]), 1200.0, atol=1.0)
            assert np.isclose(np.mean(grid._zcornsv[i, j, 6, :]), 1500.0, atol=1.0)


def test_skip_faults():
    """skip_faults=True: faulted pillars unchanged, non-faulted conformed."""
    grid = _make_faulted_grid()
    original_zcorn = grid._zcornsv.copy()
    surfs = [
        _make_flat_surface(4, 4, 100.0, 100.0, 800.0),
        _make_flat_surface(4, 4, 100.0, 100.0, 1400.0),
    ]

    grid.conform_to_surfaces(surfaces=surfs, layers_per_zone=[4], skip_faults=True)

    # Faulted pillar (i=2) unchanged
    for j in range(grid.nrow + 1):
        np.testing.assert_array_equal(
            grid._zcornsv[2, j, :, :], original_zcorn[2, j, :, :]
        )

    # Non-faulted pillars modified
    for i in [0, 1, 3]:
        for j in range(grid.nrow + 1):
            assert not np.array_equal(
                grid._zcornsv[i, j, :, :], original_zcorn[i, j, :, :]
            )


def test_pillar_outside_surface_unchanged():
    """Pillars outside the surface extent keep their original ZCORN."""
    grid = create_box_grid(
        dimension=(3, 3, 4),
        origin=(0.0, 0.0, 1000.0),
        increment=(100.0, 100.0, 50.0),
    )
    original_zcorn = grid._zcornsv.copy()

    # Surfaces cover only x=[500, 800], far from grid x=[0, 300]
    surfs = [
        _make_flat_surface(4, 4, 100.0, 100.0, 900.0, xori=500.0),
        _make_flat_surface(4, 4, 100.0, 100.0, 1300.0, xori=500.0),
    ]

    grid.conform_to_surfaces(surfaces=surfs, layers_per_zone=[4])

    np.testing.assert_array_equal(grid._zcornsv, original_zcorn)


# --- Validation ---


def test_too_few_surfaces():
    grid = create_box_grid(
        dimension=(3, 3, 3), origin=(0, 0, 0), increment=(100, 100, 10)
    )
    surf = _make_flat_surface(4, 4, 100.0, 100.0, 100.0)
    with pytest.raises(ValueError, match="At least 2 surfaces"):
        grid.conform_to_surfaces(surfaces=[surf], layers_per_zone=[])


def test_surface_count_mismatch():
    grid = create_box_grid(
        dimension=(3, 3, 3), origin=(0, 0, 0), increment=(100, 100, 10)
    )
    surfs = [_make_flat_surface(4, 4, 100, 100, d) for d in [100, 200, 300]]
    with pytest.raises(ValueError, match="Number of surfaces"):
        grid.conform_to_surfaces(surfaces=surfs, layers_per_zone=[3])


def test_layer_count_mismatch():
    grid = create_box_grid(
        dimension=(3, 3, 5), origin=(0, 0, 0), increment=(100, 100, 10)
    )
    surfs = [_make_flat_surface(4, 4, 100, 100, d) for d in [100, 200]]
    with pytest.raises(ValueError, match="Sum of layers_per_zone"):
        grid.conform_to_surfaces(surfaces=surfs, layers_per_zone=[3])


def test_zero_layers_in_zone():
    grid = create_box_grid(
        dimension=(3, 3, 3), origin=(0, 0, 0), increment=(100, 100, 10)
    )
    surfs = [_make_flat_surface(4, 4, 100, 100, d) for d in [100, 200, 300]]
    with pytest.raises(ValueError, match="layers_per_zone must be >= 1"):
        grid.conform_to_surfaces(surfaces=surfs, layers_per_zone=[0, 3])
