"""Tests for xtgeo.gridproperty_from_cube function."""

import numpy as np
import pytest

import xtgeo


def _make_cube(
    ncol=5,
    nrow=6,
    nlay=7,
    xori=0.0,
    yori=0.0,
    zori=0.0,
    xinc=25.0,
    yinc=25.0,
    zinc=4.0,
    rotation=0.0,
    yflip=1,
):
    """Helper to create a cube with known linear values: val = i + j*10 + k*100."""
    vals = np.zeros((ncol, nrow, nlay), dtype=np.float32)
    for i in range(ncol):
        for j in range(nrow):
            for k in range(nlay):
                vals[i, j, k] = float(i + j * 10 + k * 100)
    return xtgeo.Cube(
        ncol=ncol,
        nrow=nrow,
        nlay=nlay,
        xinc=xinc,
        yinc=yinc,
        zinc=zinc,
        xori=xori,
        yori=yori,
        zori=zori,
        rotation=rotation,
        yflip=yflip,
        values=vals,
    )


# --- Basic aligned tests ---


def test_simple_aligned_nearest():
    """Grid and cube perfectly aligned, nearest should give exact values."""
    cube = _make_cube(ncol=5, nrow=6, nlay=7, xinc=25.0, yinc=25.0, zinc=4.0)
    grid = xtgeo.create_box_grid(
        dimension=(5, 6, 7),
        origin=(0.0, 0.0, 0.0),
        oricenter=True,
        increment=(25.0, 25.0, 4.0),
    )
    prop = xtgeo.gridproperty_from_cube(grid, cube, name="seis")
    assert prop.name == "seis"
    assert prop.ncol == 5
    assert prop.nrow == 6
    assert prop.nlay == 7
    # Cell (0,0,0) center at (0, 0, 0) => cube index (0,0,0) => value 0
    assert prop.values[0, 0, 0] == pytest.approx(0.0, abs=1e-3)
    # Cell (1,2,3) => value 1 + 2*10 + 3*100 = 321
    assert prop.values[1, 2, 3] == pytest.approx(321.0, abs=1e-3)


def test_simple_aligned_trilinear():
    """Trilinear on aligned grid should give same result at cell centers."""
    cube = _make_cube(ncol=5, nrow=6, nlay=7, xinc=25.0, yinc=25.0, zinc=4.0)
    grid = xtgeo.create_box_grid(
        dimension=(5, 6, 7),
        origin=(0.0, 0.0, 0.0),
        oricenter=True,
        increment=(25.0, 25.0, 4.0),
    )
    prop = xtgeo.gridproperty_from_cube(
        grid, cube, name="seis_tri", interpolation="trilinear"
    )
    # Interior cell — trilinear at exact node should match nearest
    # Cell (2, 2, 2) => value 2 + 20 + 200 = 222
    assert prop.values[2, 2, 2] == pytest.approx(222.0, abs=1e-3)


def test_outside_cube_gets_fill_value():
    """Grid cells outside the cube get outside_value (default 0)."""
    cube = _make_cube(ncol=3, nrow=3, nlay=3, xinc=10.0, yinc=10.0, zinc=5.0)
    grid = xtgeo.create_box_grid(
        dimension=(10, 10, 10),
        origin=(-100.0, -100.0, -100.0),
        increment=(50.0, 50.0, 50.0),
    )
    prop = xtgeo.gridproperty_from_cube(grid, cube)
    active = prop.values.compressed()
    assert 0.0 in active


def test_outside_cube_custom_fill_value():
    """Grid cells outside the cube get a user-defined outside_value."""
    cube = _make_cube(ncol=3, nrow=3, nlay=3, xinc=10.0, yinc=10.0, zinc=5.0)
    grid = xtgeo.create_box_grid(
        dimension=(10, 10, 10),
        origin=(-100.0, -100.0, -100.0),
        increment=(50.0, 50.0, 50.0),
    )
    prop = xtgeo.gridproperty_from_cube(grid, cube, outside_value=-999.0)
    active = prop.values.compressed()
    assert -999.0 in active


def test_invalid_interpolation_raises():
    """An invalid interpolation name should raise ValueError."""
    cube = _make_cube()
    grid = xtgeo.create_box_grid((3, 3, 3))
    with pytest.raises(ValueError, match="Invalid interpolation"):
        xtgeo.gridproperty_from_cube(grid, cube, interpolation="spline")


# --- Rotated cube tests ---


def test_rotated_cube_nearest():
    """A rotated cube should still sample correctly at its own node locations."""
    cube = _make_cube(
        ncol=5, nrow=6, nlay=7, xinc=25.0, yinc=25.0, zinc=4.0, rotation=30.0
    )
    grid = xtgeo.grid_from_cube(cube, propname=None, oricenter=True)
    prop = xtgeo.gridproperty_from_cube(grid, cube, interpolation="nearest")

    assert prop.values[0, 0, 0] == pytest.approx(0.0, abs=1e-3)
    # Cell (2, 3, 4) => 2 + 30 + 400 = 432
    assert prop.values[2, 3, 4] == pytest.approx(432.0, abs=1e-3)


def test_rotated_cube_trilinear():
    """Trilinear on rotated cube at exact nodes."""
    cube = _make_cube(
        ncol=5, nrow=6, nlay=7, xinc=25.0, yinc=25.0, zinc=4.0, rotation=45.0
    )
    grid = xtgeo.grid_from_cube(cube, propname=None, oricenter=True)
    prop = xtgeo.gridproperty_from_cube(grid, cube, interpolation="trilinear")
    # Interior cell (2, 2, 2) => 222
    assert prop.values[2, 2, 2] == pytest.approx(222.0, abs=1e-3)


def test_rotated_cube_cubic():
    """Cubic on rotated cube at exact nodes should be close to exact."""
    cube = _make_cube(
        ncol=5, nrow=6, nlay=7, xinc=25.0, yinc=25.0, zinc=4.0, rotation=30.0
    )
    grid = xtgeo.grid_from_cube(cube, propname=None, oricenter=True)
    prop = xtgeo.gridproperty_from_cube(grid, cube, interpolation="cubic")
    # Interior cell (2, 3, 3) => 2 + 30 + 300 = 332
    # Cubic at exact nodes may have slight overshoot, so use wider tolerance
    assert prop.values[2, 3, 3] == pytest.approx(332.0, abs=0.5)


def test_rotated_cube_catmull_rom():
    """Catmull-Rom at exact nodes should match exactly (passes through data)."""
    cube = _make_cube(
        ncol=5, nrow=6, nlay=7, xinc=25.0, yinc=25.0, zinc=4.0, rotation=30.0
    )
    grid = xtgeo.grid_from_cube(cube, propname=None, oricenter=True)
    prop = xtgeo.gridproperty_from_cube(grid, cube, interpolation="catmull-rom")
    # Interior cell (2, 3, 3) => 2 + 30 + 300 = 332
    # Catmull-Rom passes exactly through data points
    assert prop.values[2, 3, 3] == pytest.approx(332.0, abs=1e-3)


# --- Yflip tests ---


def test_yflip_minus1_nearest():
    """Cube with yflip=-1 should be handled correctly."""
    cube = _make_cube(yflip=-1)
    grid = xtgeo.grid_from_cube(cube, propname=None, oricenter=True)
    prop = xtgeo.gridproperty_from_cube(grid, cube, interpolation="nearest")
    # Cell (1, 1, 1) => 1 + 10 + 100 = 111
    assert prop.values[1, 1, 1] == pytest.approx(111.0, abs=1e-3)


# --- Constant cube tests ---


def test_constant_cube():
    """All grid cells inside cube should have the constant value."""
    cube = xtgeo.Cube(
        ncol=10,
        nrow=10,
        nlay=10,
        xinc=10.0,
        yinc=10.0,
        zinc=2.0,
        xori=0.0,
        yori=0.0,
        zori=0.0,
        values=42.0,
    )
    grid = xtgeo.grid_from_cube(cube, propname=None, oricenter=True)
    prop = xtgeo.gridproperty_from_cube(grid, cube, interpolation="nearest")
    active_vals = prop.values.compressed()
    assert np.allclose(active_vals, 42.0)

    prop_tri = xtgeo.gridproperty_from_cube(
        grid, cube, interpolation="trilinear", outside_value=42.0
    )
    active_tri = prop_tri.values.compressed()
    assert np.allclose(active_tri, 42.0)

    prop_cub = xtgeo.gridproperty_from_cube(
        grid, cube, interpolation="cubic", outside_value=42.0
    )
    active_cub = prop_cub.values.compressed()
    assert np.allclose(active_cub, 42.0)


# --- Yflip with non-nearest interpolation ---


@pytest.mark.parametrize("interpolation", ["trilinear", "cubic", "catmull-rom"])
def test_yflip_minus1_interpolations(interpolation):
    """Cube with yflip=-1 should work for all interpolation methods."""
    cube = _make_cube(yflip=-1)
    grid = xtgeo.grid_from_cube(cube, propname=None, oricenter=True)
    prop = xtgeo.gridproperty_from_cube(grid, cube, interpolation=interpolation)
    # Interior cell (2, 2, 2) => 2 + 20 + 200 = 222
    assert prop.values[2, 2, 2] == pytest.approx(222.0, abs=0.5)


# --- Masked / inactive grid cells ---


def test_inactive_cells_no_warning():
    """Grid with inactive cells should not raise RuntimeWarning from NaN casts."""
    cube = _make_cube(ncol=5, nrow=6, nlay=7, xinc=25.0, yinc=25.0, zinc=4.0)
    grid = xtgeo.create_box_grid(
        dimension=(5, 6, 7),
        origin=(0.0, 0.0, 0.0),
        oricenter=True,
        increment=(25.0, 25.0, 4.0),
    )
    # Deactivate some cells
    actnum = grid.get_actnum()
    actnum.values[0, 0, 0] = 0
    actnum.values[2, 3, 4] = 0
    actnum.values[4, 5, 6] = 0
    grid.set_actnum(actnum)

    for method in ("nearest", "trilinear", "cubic", "catmull-rom"):
        prop = xtgeo.gridproperty_from_cube(grid, cube, interpolation=method)
        # Deactivated cells must be masked
        assert prop.values[0, 0, 0] is np.ma.masked
        assert prop.values[2, 3, 4] is np.ma.masked
        assert prop.values[4, 5, 6] is np.ma.masked
        # An active interior cell should still have the correct value
        assert prop.values[1, 1, 1] == pytest.approx(111.0, abs=0.5)
