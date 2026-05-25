"""Functions for conforming a grid's ZCORN to a set of surfaces."""

from __future__ import annotations

from typing import TYPE_CHECKING

import xtgeo._internal as _internal  # type: ignore
from xtgeo.common import null_logger

logger = null_logger(__name__)

if TYPE_CHECKING:
    from xtgeo.grid3d import Grid
    from xtgeo.surface.regular_surface import RegularSurface


def conform_grid_to_surfaces(
    grid: Grid,
    surfaces: list[RegularSurface],
    layers_per_zone: list[int],
    skip_faults: bool = False,
    tolerance: float = 1e-6,
) -> None:
    """Conform grid ZCORN to a set of surfaces."""
    n_surfaces = len(surfaces)
    n_zones = len(layers_per_zone)

    if n_surfaces < 2:
        raise ValueError("At least 2 surfaces are required (top and bottom).")

    if n_surfaces != n_zones + 1:
        raise ValueError(
            f"Number of surfaces ({n_surfaces}) must be "
            f"len(layers_per_zone) + 1 ({n_zones + 1})."
        )

    total_layers = sum(layers_per_zone)
    if total_layers != grid.nlay:
        raise ValueError(
            f"Sum of layers_per_zone ({total_layers}) must equal "
            f"grid nlay ({grid.nlay})."
        )

    if any(lpz < 1 for lpz in layers_per_zone):
        raise ValueError("All values in layers_per_zone must be >= 1.")

    grid._set_xtgformat2()

    # Convert surfaces to C++ objects for fast sampling
    cpp_surfs = [_internal.regsurf.RegularSurface(s) for s in surfaces]

    # Modify zcornsv in-place via C++ Grid method
    _internal.grid3d.Grid(grid).conform_grid_to_surfaces(
        cpp_surfs,
        list(layers_per_zone),
        skip_faults,
        tolerance,
    )

    # Set subgrids to reflect the zone structure
    subgrids = {f"zone_{i + 1}": layers_per_zone[i] for i in range(n_zones)}
    grid.set_subgrids(subgrids)

    logger.info("Grid conformed to %d surfaces with %d zones", n_surfaces, n_zones)
