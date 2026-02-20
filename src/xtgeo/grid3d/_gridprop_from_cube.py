"""Sample cube values at 3D grid cell centers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from xtgeo.cube.cube1 import Cube

    from .grid import Grid


def sample_cube_to_grid(
    grid: Grid,
    cube: Cube,
    interpolation: Literal["nearest", "trilinear", "cubic", "catmull-rom"] = "nearest",
    outside_value: float = 0.0,
) -> np.ma.MaskedArray:
    """Sample cube values at grid cell centers and return as a masked array.

    Args:
        grid: The 3D grid whose cell centers define the sampling locations.
        cube: The seismic cube to sample values from.
        interpolation: ``"nearest"``, ``"trilinear"``, ``"cubic"`` or
            ``"catmull-rom"``.
        outside_value: Value for active grid cells outside the cube extent.

    Returns:
        A masked array with shape (ncol, nrow, nlay) containing sampled values.
    """
    valid = ("nearest", "trilinear", "cubic", "catmull-rom")
    if interpolation not in valid:
        raise ValueError(
            f"Invalid interpolation method '{interpolation}'. "
            f"Supported methods are {', '.join(repr(v) for v in valid)}."
        )

    # Get grid cell center coordinates as masked arrays (ncol, nrow, nlay)
    xprop, yprop, zprop = grid.get_xyz(asmasked=True)
    xv = xprop.values
    yv = yprop.values
    zv = zprop.values

    # All three coordinate arrays share the same mask:
    combined_mask = np.asarray(xv.mask, dtype=bool)

    # Transform (x, y, z) from world coordinates to the cube's local frame,
    # accounting for rotation around the cube origin.
    angle_rad = np.deg2rad(cube.rotation)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    dx = np.asarray(xv, dtype=np.float64) - cube.xori
    dy = np.asarray(yv, dtype=np.float64) - cube.yori

    # Local coordinates along cube inline (I) and crossline (J) axes
    local_x = dx * cos_a + dy * sin_a
    local_y = (-dx * sin_a + dy * cos_a) * cube.yflip

    # Fractional indices in cube space
    fi = local_x / cube.xinc
    fj = local_y / cube.yinc
    fk = (np.asarray(zv, dtype=np.float64) - cube.zori) / cube.zinc

    if interpolation == "nearest":
        return _sample_nearest(cube, fi, fj, fk, combined_mask, outside_value)

    if interpolation == "cubic":
        return _sample_cubic(cube, fi, fj, fk, combined_mask, outside_value)

    if interpolation == "catmull-rom":
        return _sample_catmull_rom(cube, fi, fj, fk, combined_mask, outside_value)

    return _sample_trilinear(cube, fi, fj, fk, combined_mask, outside_value)


def _sample_nearest(
    cube: Cube,
    fi: np.ndarray,
    fj: np.ndarray,
    fk: np.ndarray,
    combined_mask: np.ndarray,
    outside_value: float = 0.0,
) -> np.ma.MaskedArray:
    """Nearest-neighbor sampling."""
    # Fill NaN (from masked/inactive grid cells) with 0 before int cast to
    # avoid RuntimeWarning; these cells are masked out regardless.
    ii = np.rint(np.nan_to_num(fi, nan=0.0)).astype(int)
    jj = np.rint(np.nan_to_num(fj, nan=0.0)).astype(int)
    kk = np.rint(np.nan_to_num(fk, nan=0.0)).astype(int)

    outside = (
        (ii < 0)
        | (ii >= cube.ncol)
        | (jj < 0)
        | (jj >= cube.nrow)
        | (kk < 0)
        | (kk >= cube.nlay)
    )

    ii_safe = np.clip(ii, 0, cube.ncol - 1)
    jj_safe = np.clip(jj, 0, cube.nrow - 1)
    kk_safe = np.clip(kk, 0, cube.nlay - 1)

    sampled = cube.values[ii_safe, jj_safe, kk_safe].astype(np.float64)
    sampled[outside] = outside_value

    return np.ma.MaskedArray(sampled, mask=combined_mask, dtype=np.float64)


def _sample_cubic(
    cube: Cube,
    fi: np.ndarray,
    fj: np.ndarray,
    fk: np.ndarray,
    combined_mask: np.ndarray,
    outside_value: float = 0.0,
) -> np.ma.MaskedArray:
    """Tricubic spline interpolation via scipy.ndimage.map_coordinates."""
    from scipy.ndimage import map_coordinates  # lazy import intended

    fi = np.nan_to_num(fi, nan=0.0)
    fj = np.nan_to_num(fj, nan=0.0)
    fk = np.nan_to_num(fk, nan=0.0)

    coords = np.array(
        [fi.ravel(), fj.ravel(), fk.ravel()],
        dtype=np.float64,
    )

    sampled = map_coordinates(
        cube.values.astype(np.float64),
        coords,
        order=3,
        mode="constant",
        cval=outside_value,
    ).reshape(fi.shape)

    return np.ma.MaskedArray(sampled, mask=combined_mask, dtype=np.float64)


def _catmull_rom_weights(t: np.ndarray) -> tuple[np.ndarray, ...]:
    """Catmull-Rom weights for offsets -1, 0, +1, +2 relative to floor index."""
    t2 = t * t
    t3 = t2 * t
    w0 = -0.5 * t3 + t2 - 0.5 * t
    w1 = 1.5 * t3 - 2.5 * t2 + 1.0
    w2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
    w3 = 0.5 * t3 - 0.5 * t2
    return w0, w1, w2, w3


def _sample_catmull_rom(
    cube: Cube,
    fi: np.ndarray,
    fj: np.ndarray,
    fk: np.ndarray,
    combined_mask: np.ndarray,
    outside_value: float = 0.0,
) -> np.ma.MaskedArray:
    """Tricubic Catmull-Rom interpolation (separable, vectorized).

    Uses the Catmull-Rom cardinal spline kernel applied separably along each
    axis. This passes exactly through data points and is commonly used in
    seismic interpretation tools.
    """
    fi = np.nan_to_num(fi, nan=0.0)
    fj = np.nan_to_num(fj, nan=0.0)
    fk = np.nan_to_num(fk, nan=0.0)

    i0 = np.floor(fi).astype(int)
    j0 = np.floor(fj).astype(int)
    k0 = np.floor(fk).astype(int)

    # Catmull-Rom needs indices floor-1 .. floor+2 in each dimension
    outside = (
        (i0 - 1 < 0)
        | (i0 + 2 >= cube.ncol)
        | (j0 - 1 < 0)
        | (j0 + 2 >= cube.nrow)
        | (k0 - 1 < 0)
        | (k0 + 2 >= cube.nlay)
    )

    wi = _catmull_rom_weights(fi - i0)
    wj = _catmull_rom_weights(fj - j0)
    wk = _catmull_rom_weights(fk - k0)

    nc, nr, nl = cube.ncol, cube.nrow, cube.nlay

    # Use a plain ndarray (not masked) and flat 1D indexing for speed.
    # np.ma.getdata strips the mask; np.ascontiguousarray ensures layout.
    cube_flat = np.ascontiguousarray(
        np.ma.getdata(cube.values), dtype=np.float64
    ).ravel()
    stride_i = nr * nl
    stride_j = nl

    # Pre-compute clipped k-indices (same for all i, j offsets).
    k_idx = [np.clip(k0 + dk - 1, 0, nl - 1) for dk in range(4)]

    # Separable: k first (inlined), then j, then i.
    sampled = np.zeros(fi.shape, dtype=np.float64)

    for di in range(4):
        i_base = np.clip(i0 + di - 1, 0, nc - 1) * stride_i
        acc_j = np.zeros(fi.shape, dtype=np.float64)

        for dj in range(4):
            ij_base = i_base + np.clip(j0 + dj - 1, 0, nr - 1) * stride_j
            # Interpolate along k using flat 1D look-ups.
            val_k = (
                wk[0] * cube_flat[ij_base + k_idx[0]]
                + wk[1] * cube_flat[ij_base + k_idx[1]]
                + wk[2] * cube_flat[ij_base + k_idx[2]]
                + wk[3] * cube_flat[ij_base + k_idx[3]]
            )
            acc_j += wj[dj] * val_k

        sampled += wi[di] * acc_j

    sampled[outside] = outside_value

    return np.ma.MaskedArray(sampled, mask=combined_mask, dtype=np.float64)


def _sample_trilinear(
    cube: Cube,
    fi: np.ndarray,
    fj: np.ndarray,
    fk: np.ndarray,
    combined_mask: np.ndarray,
    outside_value: float = 0.0,
) -> np.ma.MaskedArray:
    """Trilinear interpolation sampling."""
    # Fill NaN (from masked/inactive grid cells) with 0 before int cast to
    # avoid RuntimeWarning; these cells are masked out regardless.
    fi = np.nan_to_num(fi, nan=0.0)
    fj = np.nan_to_num(fj, nan=0.0)
    fk = np.nan_to_num(fk, nan=0.0)

    i0 = np.floor(fi).astype(int)
    j0 = np.floor(fj).astype(int)
    k0 = np.floor(fk).astype(int)

    di = fi - i0
    dj = fj - j0
    dk = fk - k0

    i1 = i0 + 1
    j1 = j0 + 1
    k1 = k0 + 1

    outside = (
        (i0 < 0)
        | (i1 >= cube.ncol)
        | (j0 < 0)
        | (j1 >= cube.nrow)
        | (k0 < 0)
        | (k1 >= cube.nlay)
    )

    i0s = np.clip(i0, 0, cube.ncol - 1)
    i1s = np.clip(i1, 0, cube.ncol - 1)
    j0s = np.clip(j0, 0, cube.nrow - 1)
    j1s = np.clip(j1, 0, cube.nrow - 1)
    k0s = np.clip(k0, 0, cube.nlay - 1)
    k1s = np.clip(k1, 0, cube.nlay - 1)

    cvals = cube.values
    c000 = cvals[i0s, j0s, k0s]
    c100 = cvals[i1s, j0s, k0s]
    c010 = cvals[i0s, j1s, k0s]
    c110 = cvals[i1s, j1s, k0s]
    c001 = cvals[i0s, j0s, k1s]
    c101 = cvals[i1s, j0s, k1s]
    c011 = cvals[i0s, j1s, k1s]
    c111 = cvals[i1s, j1s, k1s]

    sampled = (
        c000 * (1 - di) * (1 - dj) * (1 - dk)
        + c100 * di * (1 - dj) * (1 - dk)
        + c010 * (1 - di) * dj * (1 - dk)
        + c110 * di * dj * (1 - dk)
        + c001 * (1 - di) * (1 - dj) * dk
        + c101 * di * (1 - dj) * dk
        + c011 * (1 - di) * dj * dk
        + c111 * di * dj * dk
    )
    sampled[outside] = outside_value

    return np.ma.MaskedArray(sampled, mask=combined_mask, dtype=np.float64)
