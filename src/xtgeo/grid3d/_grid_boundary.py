from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from xtgeo.xyz.points import Points
from xtgeo.xyz.polygons import Polygons

if TYPE_CHECKING:
    from xtgeo.grid3d import Grid


def create_boundary(
    grid: Grid,
    alpha_factor: float = 1.0,
    convex: bool = False,
    simplify: bool | dict[str, Any] = True,
    filter_array: np.ndarray | None = None,
) -> Polygons:
    """Create boundary polygons for a grid."""

    xval, yval, zval = (prop.values for prop in grid.get_xyz())

    if filter_array is not None:
        if filter_array.shape != grid.dimensions:
            raise ValueError(
                "The filter_array needs to have the same dimensions as the grid. "
                f"Found: {filter_array.shape=} {grid.dimensions=}"
            )
        xval = np.ma.masked_where(~filter_array, xval)
        yval = np.ma.masked_where(~filter_array, yval)
        zval = np.ma.masked_where(~filter_array, zval)

    # for performance create average points along layers
    xval = np.ma.mean(xval, axis=2)
    yval = np.ma.mean(yval, axis=2)
    zval = np.ma.mean(zval, axis=2)

    xyz_values = np.column_stack(
        (
            xval[~xval.mask].ravel(),
            yval[~yval.mask].ravel(),
            zval[~zval.mask].ravel(),
        )
    )

    pol = Polygons.boundary_from_points(
        points=Points(xyz_values),
        alpha_factor=alpha_factor,
        alpha=_estimate_alpha_for_grid(grid),
        convex=convex,
    )

    if simplify:
        if isinstance(simplify, bool):
            pol.simplify(tolerance=0.1)
        elif isinstance(simplify, dict) and "tolerance" in simplify:
            pol.simplify(**simplify)
        else:
            raise ValueError("Invalid values for simplify keyword")

    return pol


def _estimate_alpha_for_grid(grid: Grid) -> float:
    """
    Estimate an alpha based on grid resolution.
    Max dx and dy is used as basis for calculation to ensure that the alpha
    computed is always high enough to prevent polygons appearing around areas
    of the grid where cells have larger than average dx/dy increments.
    """
    dx, dy = grid.get_dx(), grid.get_dy()
    xinc, yinc = dx.values.max(), dy.values.max()
    return math.ceil(math.sqrt(xinc**2 + yinc**2) / 2)
