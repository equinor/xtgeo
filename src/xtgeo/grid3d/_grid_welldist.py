"""Private module for grid cell distances to wells."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.spatial import cKDTree

from xtgeo.well import BlockedWell, Well
from xtgeo.xyz import Points

from .grid_property import GridProperty

if TYPE_CHECKING:
    from .grid import Grid


def get_distance_to_wells(
    grid: Grid,
    wells: Sequence[Well],
    metric: Literal["euclid", "horizontal"] = "euclid",
    name: str = "DISTANCE_WELL",
) -> GridProperty:
    """Get the distance from each active cell center to the nearest well."""
    if metric not in ("euclid", "horizontal"):
        raise ValueError(f"Unsupported metric {metric!r}; use 'euclid' or 'horizontal'")

    if (
        not wells
        or not isinstance(wells, Sequence)
        or not all(isinstance(well, Well) for well in wells)
    ):
        raise ValueError("wells must be a non-empty sequence of Wells/BlockedWells")

    cell_indices = [_get_cell_indices(grid, well) for well in wells]

    cell_indices = [indices for indices in cell_indices if len(indices)]
    if not cell_indices:
        raise ValueError("None of the wells penetrates the grid")

    indices = np.unique(np.concatenate(cell_indices), axis=0)
    xprop, yprop, zprop = grid.get_xyz(asmasked=True)
    actnum = grid.get_actnum().values == 1
    centers = np.column_stack(
        (xprop.values[actnum], yprop.values[actnum], zprop.values[actnum])
    )
    ijk_indices = tuple(indices.T)
    targets = np.column_stack(
        (
            np.ma.filled(xprop.values[ijk_indices], np.nan),
            np.ma.filled(yprop.values[ijk_indices], np.nan),
            np.ma.filled(zprop.values[ijk_indices], np.nan),
        )
    )
    targets = targets[np.all(np.isfinite(targets), axis=1)]
    if not len(targets):
        raise ValueError("None of the wells penetrates an active grid cell")

    dimensions = 2 if metric == "horizontal" else 3
    values = np.ma.masked_all(grid.dimensions, dtype=float)
    valid_centers = np.all(np.isfinite(centers), axis=1)
    centers = centers[valid_centers]
    # OMP_NUM_THREADS is an environment variable used in OpenMP programs
    # to set the number of threads used for parallel region
    # If OMP_NUM_THREADS is not set, cKDTree will use all available CPUs.
    ncpu = int(os.environ.get("OMP_NUM_THREADS", -1))
    distances = cKDTree(targets[:, :dimensions]).query(
        centers[:, :dimensions], workers=ncpu
    )[0]
    active_indices = np.flatnonzero(actnum)
    values.flat[active_indices[valid_centers]] = distances

    return GridProperty(
        ncol=grid.ncol,
        nrow=grid.nrow,
        nlay=grid.nlay,
        values=values,
        name=name,
        discrete=False,
    )


def _get_cell_indices(grid: Grid, well: Well) -> np.ndarray:
    """Get zero-based, in-grid cell indices penetrated by a well."""
    if isinstance(well, BlockedWell):
        dataframe = well.get_dataframe(copy=False)
        columns = ("I_INDEX", "J_INDEX", "K_INDEX")
        if not set(columns).issubset(dataframe.columns):
            raise ValueError("BlockedWell is missing I_INDEX, J_INDEX, or K_INDEX")
        indices = dataframe.loc[:, columns].dropna().to_numpy(dtype=int)
    else:
        dataframe = well.get_dataframe(copy=False)
        points = Points(
            values=dataframe.loc[:, [well.xname, well.yname, well.zname]].dropna(),
            xname=well.xname,
            yname=well.yname,
            zname=well.zname,
        )
        indices = np.asarray(
            grid.get_ijk_from_points(
                points,
                zerobased=True,
                dataframe=True,
                includepoints=False,
            )
        )
    in_grid = np.all(indices >= 0, axis=1) & np.all(
        indices < np.asarray(grid.dimensions), axis=1
    )
    return indices[in_grid]
