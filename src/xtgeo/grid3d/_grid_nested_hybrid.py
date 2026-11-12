"""Detect the nested hybrid layout produced by :func:`xtgeo.grid_merge`.

A nested hybrid grid has a coarse block on the left (low I) and a refined block on
the right, separated by fully inactive I-columns. The detection is geometric: all
active cell centres of the refined block must lie inside the XY footprint of the
coarse block, including its inactive (replaced) cells.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import shapely

if TYPE_CHECKING:
    from xtgeo.grid3d import Grid


@dataclass(frozen=True)
class NestedHybridBlock:
    """Location of one block of a nested hybrid grid.

    Attributes:
        istart: Zero-based first I-column of the block.
        istop: Zero-based I-column after the last one (exclusive).
    """

    istart: int
    istop: int


def detect_nested_hybrid(grid: Grid) -> list[NestedHybridBlock]:
    """Return the blocks of a coarse-left, refined-right nested hybrid grid.

    The result is empty if the grid is not a nested hybrid. Only the first refined
    block is verified: it is rejected cheaply if its pillars cannot overlap the
    coarse bounds, otherwise every active refined cell centre must fall inside the
    coarse footprint. Further blocks are located but not verified.
    """
    grid._set_xtgformat2()
    blocks = _find_blocks(grid._actnumsv)
    if len(blocks) < 2:
        return []

    gap_start, gap_end, fine_end = blocks[0].istop, blocks[1].istart, blocks[1].istop
    # Geometry includes one more pillar than cell columns.
    fine_coords = grid._coordsv[gap_end : fine_end + 1]
    fine_zcorn = grid._zcornsv[gap_end : fine_end + 1]
    active = grid._actnumsv[gap_end:fine_end] > 0
    envelope = _active_envelope(fine_coords, fine_zcorn, active)
    depth_range = (envelope[0][:, 2].min(), envelope[1][:, 2].max())

    coarse = _coarse_cells(
        grid._coordsv[: gap_start + 1],
        grid._zcornsv[: gap_start + 1],
        depth_range,
        np.any(grid._actnumsv[:gap_start] > 0, axis=2),
    )
    if coarse.size == 0:
        return []
    lower, upper = coarse.reshape(-1, 2).min(axis=0), coarse.reshape(-1, 2).max(axis=0)

    # Cheap rejection of disjoint blocks before any per-cell work: every centre
    # lies between the active corners of the touched pillars.
    xy = np.concatenate([corners[:, :2] for corners in envelope])
    if np.any(xy.max(axis=0) < lower) or np.any(xy.min(axis=0) > upper):
        return []

    px, py = _active_xy(fine_coords, fine_zcorn, active)
    inside_bounds = (
        (px >= lower[0]) & (px <= upper[0]) & (py >= lower[1]) & (py <= upper[1])
    )
    if not inside_bounds.all():
        return []

    # Bounds alone would also accept points in concave recesses of the coarse grid.
    footprint = _coarse_footprint(coarse, px, py)
    if not shapely.contains_xy(footprint, px, py).all():
        return []
    return blocks


def _active_envelope(
    coordsv: np.ndarray, zcornsv: np.ndarray, active: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the shallowest and deepest active corner of every touched pillar.

    Both results have shape (n, 3), holding X, Y and Z. Only corners of active cells
    count, so merge padding and inactive cells do not contribute.
    """
    shape = (active.shape[0] + 1, active.shape[1] + 1)
    top, base = np.full(shape, np.inf), np.full(shape, -np.inf)
    for di, dj, quadrant in [(0, 0, 3), (1, 0, 2), (1, 1, 0), (0, 1, 1)]:
        corners = zcornsv[
            di : di + active.shape[0], dj : dj + active.shape[1], :, quadrant
        ]
        window = (slice(di, di + active.shape[0]), slice(dj, dj + active.shape[1]))
        top[window] = np.minimum(
            top[window],
            np.min(corners[..., :-1], axis=2, where=active, initial=np.inf),
        )
        base[window] = np.maximum(
            base[window],
            np.max(corners[..., 1:], axis=2, where=active, initial=-np.inf),
        )
    touched = np.isfinite(top)
    pillars = coordsv[touched]
    return tuple(  # type: ignore[return-value]
        np.column_stack([_xy_at_depth(pillars, depth[touched]), depth[touched]])
        for depth in (top, base)
    )


def _xy_at_depth(pillars: np.ndarray, depths: np.ndarray | float) -> np.ndarray:
    """Return pillar XY at the given depths, even beyond the pillar endpoints.

    Collapsed pillars use their top XY, matching the C++ corner evaluator.
    """
    top = pillars[..., :3]
    direction = pillars[..., 3:] - top
    fraction = np.zeros(pillars.shape[:-1])
    np.divide(
        depths - top[..., 2],
        direction[..., 2],
        out=fraction,
        where=np.abs(direction[..., 2]) >= np.finfo(float).eps,
    )
    return top[..., :2] + fraction[..., None] * direction[..., :2]


def _active_xy(
    coordsv: np.ndarray, zcornsv: np.ndarray, active: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return X and Y of the centres of active cells, as the mean of 8 corners."""
    i, j, k = np.nonzero(active)
    xy = np.zeros((i.size, 2))
    for di, dj, quadrant in [(0, 0, 3), (1, 0, 2), (1, 1, 0), (0, 1, 1)]:
        for dk in (0, 1):
            depths = zcornsv[i + di, j + dj, k + dk, quadrant]
            xy += _xy_at_depth(coordsv[i + di, j + dj], depths)
    xy /= 8
    return xy[:, 0], xy[:, 1]


def _find_blocks(actnumsv: np.ndarray) -> list[NestedHybridBlock]:
    """Locate the blocks of a :func:`xtgeo.grid_merge` result as I-column ranges.

    ``grid_merge`` places the coarse grid first, then fully inactive I-columns
    (the separator), then the refined grid, and so on for further refinements. The
    blocks are therefore the runs of columns with any active cell, in I order.

    Returns:
        The coarse block, then each refined block. Fewer than two blocks means the
        pattern is not present.
    """
    active = np.any(actnumsv > 0, axis=(1, 2))
    edges = np.flatnonzero(np.diff(np.pad(active, 1).astype(np.int8)))
    return [
        NestedHybridBlock(int(start), int(stop)) for start, stop in edges.reshape(-1, 2)
    ]


def _coarse_cells(
    coordsv: np.ndarray,
    zcornsv: np.ndarray,
    depth_range: tuple[float, float],
    active: np.ndarray,
) -> np.ndarray:
    """Return the top/base XY corners of each valid coarse column, shape (n, 8, 2).

    Replaced (inactive, collapsed) columns are evaluated over ``depth_range`` so they
    cover the refined cells; merge padding rows are dropped.
    """
    # Approximate each column by its top/base envelope; twisted faces can differ.
    zcornsv = zcornsv[:, :, [0, -1]]
    # Walk around each column: (i,j), (i+1,j), (i+1,j+1), (i,j+1).
    # ZCORN quadrants 3, 2, 0, 1 select the depths belonging to that column.
    columns = (
        (coordsv[:-1, :-1], zcornsv[:-1, :-1, :, 3]),
        (coordsv[1:, :-1], zcornsv[1:, :-1, :, 2]),
        (coordsv[1:, 1:], zcornsv[1:, 1:, :, 0]),
        (coordsv[:-1, 1:], zcornsv[:-1, 1:, :, 1]),
    )
    # Only inactive collapsed columns are replacement candidates; active ones
    # keep their actual depths.
    collapsed = (
        np.logical_and.reduce(
            [depths[:, :, 0] == depths[:, :, 1] for _, depths in columns]
        )
        & ~active
    ).ravel()
    # grid_merge pads the rows after the last coarse row with inactive cells whose
    # pillars collapse to one point; they carry no real geometry.
    degenerate_rows = np.all(coordsv == coordsv[:1], axis=(0, 2))
    trailing = np.flip(np.logical_and.accumulate(np.flip(~np.any(active, axis=0))))
    padding_rows = (degenerate_rows[:-1] | degenerate_rows[1:]) & trailing
    valid_cells = (~padding_rows[None, :] | active).ravel()
    corners = []
    for pillars, depths in columns:
        pillars = pillars.reshape(-1, 6)
        depths = depths.reshape(-1, 2).astype(np.float64)
        # Removed coarse columns may be flattened far above their replacement cells.
        depths[collapsed] = depth_range
        corners.append(
            np.stack(
                [
                    _xy_at_depth(pillars, depths[:, 0]),
                    _xy_at_depth(pillars, depths[:, 1]),
                ],
                axis=1,
            )
        )
    # Each perimeter corner contributes a top/base pair: eight XY points per column.
    cells = np.stack(corners, axis=1).reshape(-1, 8, 2)
    return cells[valid_cells]


def _coarse_footprint(
    cells: np.ndarray, px: np.ndarray, py: np.ndarray
) -> shapely.Geometry:
    """Return the XY area swept by the coarse columns near the refined centres.

    Each column contributes its top and base quads plus side faces when inclined,
    so concave recesses of the coarse grid stay outside the footprint.
    """
    # Columns outside the fine-centre bounds cannot contain any of those centres.
    overlaps = np.all(cells.max(axis=1) >= (px.min(), py.min()), axis=1) & np.all(
        cells.min(axis=1) <= (px.max(), py.max()), axis=1
    )
    cells = cells[overlaps]
    top, base = cells[:, 0::2], cells[:, 1::2]
    inclined = np.any(top != base, axis=(1, 2))
    faces = np.stack(
        [
            top,
            base,
            np.stack((top[:, 0], top[:, 1], base[:, 1], base[:, 0]), axis=1),
            np.stack((top[:, 1], top[:, 2], base[:, 2], base[:, 1]), axis=1),
            np.stack((top[:, 2], top[:, 3], base[:, 3], base[:, 2]), axis=1),
            np.stack((top[:, 3], top[:, 0], base[:, 0], base[:, 3]), axis=1),
        ],
        axis=1,
    )
    polygons = np.concatenate(
        [
            shapely.polygons(top[~inclined]),
            shapely.polygons(faces[inclined].reshape(-1, 4, 2)),
        ]
    )
    # Top quads can be self-intersecting or have zero area.
    invalid = ~shapely.is_valid(polygons)
    polygons[invalid] = shapely.make_valid(polygons[invalid])
    return shapely.union_all(polygons[shapely.area(polygons) > 0])
