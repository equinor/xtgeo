"""Detect the nested hybrid layout produced by :func:`xtgeo.grid_merge`.

A nested hybrid grid has a coarse block on the left (low I) and a refined block on
the right, separated by fully inactive I-columns. The detection is geometric: all
active cell centres of the refined block must lie inside the XY footprint of the
coarse block, including its inactive (replaced) cells.

Each refined cell is then mapped to its parent cell in the coarse block, also from
the geometry only, see :func:`map_parents`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
import shapely
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    from collections.abc import Iterator

    from xtgeo.grid3d import Grid

_PILLAR_TOLERANCE = 1e-4
# Grid files often store coordinates as float32, which rounds large UTM coordinates
# to steps of up to 0.5 m.
_ROUNDING_TOLERANCE = float(np.finfo(np.float32).eps)
# ZCORN is stored as float32, and refinement interpolates depths.
_DEPTH_TOLERANCE = 8 * float(np.finfo(np.float32).eps)
_CHUNK_SIZE = 1 << 20


def raise_if_nested_hybrid(grid: Grid, operation: str) -> None:
    """Raise NotImplementedError if the grid is a nested hybrid grid."""
    if grid.nestedhybrid:
        raise NotImplementedError(
            f"{operation} is not supported for nested hybrid grids, "
            "see https://github.com/equinor/xtgeo/issues/1853"
        )


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
    offset = depths - top[..., 2]
    fraction = np.zeros_like(offset, dtype=float)
    np.divide(
        offset,
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


class NestedHybridCellType(IntEnum):
    """Kind of a cell in a nested hybrid grid.

    ``REPLACED_PARENT`` is an inactive coarse cell that is the parent of active
    refined cells, with a known parent layer. ``UNVERIFIED_PARENT`` is an inactive
    coarse cell with thickness in the parent column of active refined cells that
    have no known parent layer, for example because the parent cells were
    flattened. ``OTHER_INACTIVE`` covers all other inactive cells, such as gaps,
    padding, the columns between blocks and the inactive cells of refined blocks.
    """

    OTHER_INACTIVE = 0
    ACTIVE_COARSE = 1
    ACTIVE_REFINED = 2
    REPLACED_PARENT = 3
    UNVERIFIED_PARENT = 4


@dataclass(frozen=True)
class NestedHybridParents:
    """Parent cells and cell types of a nested hybrid grid.

    A parent is always a cell of the coarse block. All arrays have the grid shape
    (ncol, nrow, nlay) and are read-only.

    Attributes:
        i: I index of each cell's parent cell, zero-based and in the merged grid.
            -1 where the cell has no known parent: cells in the coarse block, in
            the columns between blocks, or where the pillars do not match a
            coarse column.
        j: J index of each cell's parent cell, zero-based and in the merged grid.
            -1 where the parent is unknown or the cell has no parent.
        k: K index of each cell's parent cell, zero-based and in the merged grid.
            -1 where the parent is unknown or the cell has no parent. K can be -1
            where I and J are known, because the parent column comes from the
            pillars but the parent layer also needs matching depths.
        celltype: The :class:`NestedHybridCellType` value of each cell.
    """

    i: np.ndarray
    j: np.ndarray
    k: np.ndarray
    celltype: np.ndarray


def map_parents(grid: Grid, blocks: list[NestedHybridBlock]) -> NestedHybridParents:
    """Map each cell of the refined blocks to its parent cell in the coarse block.

    The mapping uses the geometry and ACTNUM only. It is made for grids from
    :meth:`xtgeo.Grid.refine` and :func:`xtgeo.grid_merge`, as fmu-tools makes
    them. A cell that does not pass the checks below gets -1.

    A parent is always a cell of the coarse block, also when a block was made by
    refining a part of another refined block. The merged grid does not record
    which block a refined block came from: refining a part of a refined block
    again can give the same arrays as refining the coarse cells below it
    directly. The coarse cells are the same in both cases. The inactive cells of
    a refined block are therefore never parents.

    The coarse block is all I-columns up to the end of ``blocks[0]``, as in
    :func:`detect_nested_hybrid`. Its columns before the first active column can
    hold replaced parents too.

    Parent column (I and J):

    - The refined pillars that equal a coarse pillar give the pillar lines of the
      block that match a pillar line of the coarse block. The block gets no
      parents if a line matches more than one line.
    - A refined column must lie between two matched lines that are neighbours in
      the coarse block, in both directions. Where a block ends inside a coarse
      column, its columns have no matched line on that side and get no parent.
    - The four pillars of the refined column must equal the pillars interpolated
      from the four pillars of the coarse column. The interpolation takes the
      refined columns of one coarse column as equally wide, as one refinement
      makes them. Parts of a coarse column that were refined again with different
      factors do not pass.

    Parent layer (K), for a cell with a parent column:

    - The coarse depths are interpolated to the four corners of the refined
      column. At all four corners, the cell must lie inside the same single
      coarse layer. The cell must be thicker than the rounding of float32 depths,
      and the coarse and the refined depths of the column must never decrease.
    - On inclined pillars, the interpolated top and base of that layer must also
      meet the refined pillars.
    - Every refined column of the coarse column must pass the pillar check. In
      each of them, the cells with this parent must fill the layer from top to
      base. Only zero-thickness layers, such as merge padding, can lie between
      them.
    """
    grid._set_xtgformat2()
    coordsv, zcornsv, actnumsv = grid._coordsv, grid._zcornsv, grid._actnumsv
    parents = np.full((3, *actnumsv.shape), -1, dtype=np.int32)

    coarse = blocks[0]
    for block in blocks[1:]:
        columns = _parent_columns(coordsv, coarse, block)
        if columns is None:
            continue
        (column_i, column_j), fractions, complete = columns
        window = slice(block.istart, block.istop)
        parents[0, window] = column_i[..., None]
        parents[1, window] = column_j[..., None]
        parents[2, window] = _parent_layers(
            coordsv, zcornsv, block, (column_i, column_j), fractions, complete
        )

    celltype = _cell_types(actnumsv, zcornsv, coarse, parents)
    parents.flags.writeable = False
    celltype.flags.writeable = False
    return NestedHybridParents(parents[0], parents[1], parents[2], celltype)


def _parent_columns(
    coordsv: np.ndarray, coarse: NestedHybridBlock, block: NestedHybridBlock
) -> (
    tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], np.ndarray]
    | None
):
    """Return the coarse column of each column of a refined block.

    Returns:
        The parent I and J arrays in the merged grid, shape (ncol, nrow) of the
        block and -1 where unknown. Then the fractions of the west and east
        pillars of each block column within its parent column, shape (ncol, 2),
        and the same for the south and north pillars, shape (nrow, 2). Last, True
        where every block column of the same parent column has valid pillars,
        which is needed to check that the parent is filled. None if the block
        shares no pillars with the coarse block, or the shared pillars are not
        consistent.
    """
    child = coordsv[block.istart : block.istop + 1]
    # The coarse grid starts at the first I-column of the merged grid. Its columns
    # before the first active one can hold parents too.
    parent = coordsv[: coarse.istop + 1]
    child_i, child_j, parent_i, parent_j, spacing = _match_pillars(child, parent)
    if child_i.size == 0:
        return None
    lines_i = _lines(child_i, parent_i, child.shape[0])
    lines_j = _lines(child_j, parent_j, child.shape[1])
    if lines_i is None or lines_j is None:
        return None
    span_i, fraction_i = _spans(lines_i)
    span_j, fraction_j = _spans(lines_j)
    known = (span_i[:, None] >= 0) & (span_j[None, :] >= 0)
    column_i = np.where(known, span_i[:, None], -1)
    column_j = np.where(known, span_j[None, :], -1)

    inside = _pillars_between(
        child, coordsv, (column_i, column_j), (fraction_i, fraction_j), spacing
    )
    key = column_i.astype(np.int64) * coordsv.shape[1] + column_j
    complete = ~np.isin(key, key[(column_i >= 0) & ~inside])
    column_i[~inside] = -1
    column_j[~inside] = -1
    return (column_i, column_j), (fraction_i, fraction_j), complete


def _used_pillars(pillars: np.ndarray) -> np.ndarray:
    """Return True for pillars at a corner of a cell with real geometry.

    Such a cell has four distinct pillars and a positive area at the top or base,
    which leaves out the collapsed pillars that :func:`xtgeo.grid_merge` uses as
    padding.
    """
    corners = (pillars[:-1, :-1], pillars[1:, :-1], pillars[1:, 1:], pillars[:-1, 1:])
    distinct = np.logical_and.reduce(
        [
            np.any(first != second, axis=-1)
            for number, first in enumerate(corners)
            for second in corners[number + 1 :]
        ]
    )
    sw, se, ne, nw = corners
    diagonal_a, diagonal_b = ne - sw, nw - se
    area = [
        diagonal_a[..., x] * diagonal_b[..., y]
        - diagonal_a[..., y] * diagonal_b[..., x]
        for x, y in ((0, 1), (3, 4))
    ]
    real = distinct & ((area[0] != 0) | (area[1] != 0))
    used = np.zeros(pillars.shape[:2], dtype=bool)
    for di, dj in [(0, 0), (1, 0), (1, 1), (0, 1)]:
        used[di : di + real.shape[0], dj : dj + real.shape[1]] |= real
    return used


def _match_pillars(
    child: np.ndarray, parent: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Return the child pillars that equal a parent pillar.

    Returns:
        Child I and J pillar indices, the matching parent I and J pillar indices
        (local to each block), and the typical parent pillar spacing.
    """
    empty = np.empty(0, dtype=np.int64)
    child_used, parent_used = _used_pillars(child), _used_pillars(parent)
    spacing = _pillar_spacing(parent, parent_used)
    if spacing == 0 or not child_used.any():
        return empty, empty, empty, empty, spacing
    tree = cKDTree(parent[parent_used])
    distance, found = tree.query(
        child[child_used], distance_upper_bound=_PILLAR_TOLERANCE * spacing
    )
    hit = np.isfinite(distance)
    child_i, child_j = np.nonzero(child_used)
    parent_i, parent_j = np.nonzero(parent_used)
    found = found[hit]
    return child_i[hit], child_j[hit], parent_i[found], parent_j[found], spacing


def _pillar_spacing(pillars: np.ndarray, used: np.ndarray) -> float:
    """Return the median top XY distance between neighbouring used pillars."""
    neighbours = (used[:-1] & used[1:], used[:, :-1] & used[:, 1:])
    steps = [
        np.linalg.norm(np.diff(pillars[..., :2], axis=axis), axis=-1)[both]
        for axis, both in enumerate(neighbours)
    ]
    values = np.concatenate(steps)
    values = values[values > 0]
    return float(np.median(values)) if values.size else 0.0


def _lines(
    child_index: np.ndarray, parent_index: np.ndarray, size: int
) -> np.ndarray | None:
    """Return the parent pillar line of each child pillar line, -1 if none.

    None if a child line matches more than one parent line, or the reverse.
    """
    pairs = np.unique(np.column_stack((child_index, parent_index)), axis=0)
    if any(np.unique(lines).size < len(pairs) for lines in pairs.T):
        return None
    lines = np.full(size, -1, dtype=np.int64)
    lines[pairs[:, 0]] = pairs[:, 1]
    return lines


def _spans(lines: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the parent column of each child column along one axis.

    A child column belongs to a parent column when the nearest matched pillar lines
    on both sides match two neighbouring parent lines.

    Returns:
        The parent column, -1 if unknown, and the fractions of the two pillars of
        each child column between the matched lines, shape (ncells, 2).
    """
    cells = np.arange(len(lines) - 1)
    known = np.flatnonzero(lines >= 0)
    if known.size < 2:
        return np.full(cells.size, -1), np.zeros((cells.size, 2))
    position = np.searchsorted(known, cells, side="right") - 1
    lower = known[np.clip(position, 0, known.size - 1)]
    upper = known[np.clip(position + 1, 0, known.size - 1)]
    valid = (position >= 0) & (position + 1 < known.size)
    valid &= lines[upper] - lines[lower] == 1
    width = np.where(valid, upper - lower, 1)
    fractions = np.column_stack(((cells - lower) / width, (cells + 1 - lower) / width))
    return np.where(valid, lines[lower], -1), fractions


def _bilinear_weights(fraction_i: np.ndarray, fraction_j: np.ndarray) -> np.ndarray:
    """Return weights of the SW, SE, NW and NE parent corners for each child corner.

    Args:
        fraction_i: Fractions of the west and east child pillars, shape (n, 2).
        fraction_j: Fractions of the south and north child pillars, shape (n, 2).

    Returns:
        Shape (n, 4, 4): child corner SW, SE, NW, NE, then parent corner.
    """
    u = fraction_i[:, [0, 1, 0, 1]]
    v = fraction_j[:, [0, 0, 1, 1]]
    return np.stack([(1 - u) * (1 - v), u * (1 - v), (1 - u) * v, u * v], axis=-1)


def _pillars_between(
    child: np.ndarray,
    coordsv: np.ndarray,
    columns: tuple[np.ndarray, np.ndarray],
    fractions: tuple[np.ndarray, np.ndarray],
    spacing: float,
) -> np.ndarray:
    """Return True where the child column pillars interpolate the parent pillars."""
    column_i, column_j = columns
    inside = np.zeros(column_i.shape, dtype=bool)
    a, b = np.nonzero(column_i >= 0)
    if a.size == 0:
        return inside
    pi, pj = column_i[a, b], column_j[a, b]
    parent = _column_pillars(coordsv, pi, pj)
    expected = np.einsum(
        "ncp,npx->ncx", _bilinear_weights(fractions[0][a], fractions[1][b]), parent
    )
    actual = _column_pillars(child, a, b)
    error = np.abs(actual - expected)
    limit = _PILLAR_TOLERANCE * spacing + _ROUNDING_TOLERANCE * np.abs(expected)
    inside[a, b] = np.all(error <= limit, axis=(1, 2))
    return inside


def _column_pillars(coordsv: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Return the SW, SE, NW and NE pillars of each column."""
    return np.stack(
        [coordsv[i, j], coordsv[i + 1, j], coordsv[i, j + 1], coordsv[i + 1, j + 1]],
        axis=1,
    )


def _column_bounds(zcornsv: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Return layer boundary depths at the SW, SE, NW and NE corners of columns.

    Returns:
        Shape (n, 4, nlay + 1).
    """
    return np.stack(
        [
            zcornsv[i, j, :, 3],
            zcornsv[i + 1, j, :, 2],
            zcornsv[i, j + 1, :, 1],
            zcornsv[i + 1, j + 1, :, 0],
        ],
        axis=1,
    ).astype(np.float64)


def _parent_layers(
    coordsv: np.ndarray,
    zcornsv: np.ndarray,
    block: NestedHybridBlock,
    columns: tuple[np.ndarray, np.ndarray],
    fractions: tuple[np.ndarray, np.ndarray],
    complete: np.ndarray,
) -> np.ndarray:
    """Return the parent layer of each cell of a refined block, -1 where unknown.

    The result has shape (ncol, nrow, nlay) of the block. See :func:`map_parents`
    for the checks.
    """
    column_i, column_j = columns
    window = slice(block.istart, block.istop + 1)
    child_coordsv, child_zcornsv = coordsv[window], zcornsv[window]
    nlay = child_zcornsv.shape[2] - 1
    result = np.full((*column_i.shape, nlay), -1, dtype=np.int32)
    a, b = np.nonzero(column_i >= 0)
    # Keep the child columns of one parent column together for the fill check.
    key = column_i[a, b].astype(np.int64) * zcornsv.shape[1] + column_j[a, b]
    order = np.argsort(key, kind="stable")
    a, b, key = a[order], b[order], key[order]
    size = max(1, _CHUNK_SIZE // (4 * (nlay + 1)))
    for chunk in _chunks(key, size):
        ca, cb = a[chunk], b[chunk]
        pi, pj = column_i[ca, cb], column_j[ca, cb]
        child = _column_bounds(child_zcornsv, ca, cb)
        corners = _column_bounds(zcornsv, pi, pj)
        weights = _bilinear_weights(fractions[0][ca], fractions[1][cb])
        parent = np.einsum("ncp,npl->ncl", weights, corners)
        # Weighted sums of boundaries that never decrease do not decrease either.
        increasing = np.all(np.diff(corners, axis=-1) >= 0, axis=(1, 2))
        tolerance = _depth_tolerance(child)
        layers = _containing_layers(child, parent, increasing, tolerance)
        layers[~complete[ca, cb]] = -1
        if not np.any(layers >= 0):
            continue
        faces = _matching_parent_faces(
            _column_pillars(child_coordsv, ca, cb),
            _column_pillars(coordsv, pi, pj),
            corners,
            parent,
            weights,
        )
        top = np.maximum(layers, 0)
        top_matches = np.take_along_axis(faces, top, axis=1)
        base_matches = np.take_along_axis(faces, top + 1, axis=1)
        layers[~(top_matches & base_matches)] = -1
        result[ca, cb] = _require_fill(layers, child, parent, key[chunk], tolerance)
    return result


def _matching_parent_faces(
    child_pillars: np.ndarray,
    parent_pillars: np.ndarray,
    corner_depths: np.ndarray,
    face_depths: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Return True where an interpolated parent face meets all child pillars."""
    # The XY positions of vertical pillars were checked in _pillars_between.
    vertical = np.all(child_pillars[..., :2] == child_pillars[..., 3:5], axis=(1, 2))
    vertical &= np.all(parent_pillars[..., :2] == parent_pillars[..., 3:5], axis=(1, 2))
    if vertical.all():
        return np.ones_like(face_depths[:, 0], dtype=bool)

    # Interpolating depth and projecting onto an inclined pillar need not give
    # the same XY as interpolating the actual XYZ corners of the parent face.
    parent_xy = _xy_at_depth(parent_pillars[:, :, None], corner_depths)
    expected = np.einsum("ncp,nplx->nclx", weights, parent_xy)
    actual = _xy_at_depth(child_pillars[:, :, None], face_depths)
    scale = np.maximum(
        np.abs(parent_pillars[..., :2]), np.abs(parent_pillars[..., 3:5])
    )
    limit = _ROUNDING_TOLERANCE * np.maximum(1.0, scale.max(axis=1))
    matches = np.all(np.abs(actual - expected) <= limit[:, None, None, :], axis=(1, 3))
    return matches | vertical[:, None]


def _chunks(key: np.ndarray, size: int) -> Iterator[slice]:
    """Yield slices of about ``size`` items of sorted keys, not splitting equal keys."""
    start = 0
    while start < key.size:
        stop = min(start + size, key.size)
        stop = int(np.searchsorted(key, key[stop - 1], side="right"))
        yield slice(start, stop)
        start = stop


def _depth_tolerance(bounds: np.ndarray) -> np.ndarray:
    """Return the depth tolerance of each cell, from the cell's own depths.

    Args:
        bounds: Layer boundaries per column and corner, shape (n, 4, nlay + 1).

    Returns:
        Shape (n, nlay).
    """
    scale = np.abs(bounds).max(axis=1)
    return _DEPTH_TOLERANCE * np.maximum(1.0, np.maximum(scale[:, :-1], scale[:, 1:]))


def _containing_layers(
    child: np.ndarray,
    parent: np.ndarray,
    increasing: np.ndarray,
    tolerance: np.ndarray,
) -> np.ndarray:
    """Return the parent layer that contains each child cell at all corners.

    Args:
        child: Child layer boundaries per column and corner, shape (n, 4, nlay + 1).
        parent: Parent layer boundaries at the same corners, shape (n, 4, L + 1).
        increasing: True where the parent boundaries never decrease, shape (n,).
        tolerance: Depth tolerance of each child cell, shape (n, nlay).

    Returns:
        Shape (n, nlay), -1 where no single parent layer contains the cell.
    """
    nbounds = parent.shape[2]
    top, base = child[..., :-1], child[..., 1:]
    thickness = base - top
    # A cell thicker than twice the tolerance at one corner fits in at most one
    # layer there, because the parent boundaries never decrease.
    corner = thickness.argmax(axis=1)[:, None, :]
    thick = np.take_along_axis(thickness, corner, axis=1)[:, 0]
    middle = (
        np.take_along_axis(top, corner, axis=1)
        + np.take_along_axis(base, corner, axis=1)
    )[:, 0] / 2
    count = _count_not_above(
        np.maximum.accumulate(parent, axis=-1), corner[:, 0], middle
    )
    layer = count - 1

    valid = (thick > 2 * tolerance) & (layer >= 0) & (layer < nbounds - 1)
    valid &= increasing[:, None]
    valid &= np.all(thickness >= 0, axis=(1, 2))[:, None]
    layer = np.clip(layer, 0, nbounds - 2)
    index = np.broadcast_to(layer[:, None, :], top.shape)
    margin = tolerance[:, None, :]
    valid &= np.all(top >= np.take_along_axis(parent, index, axis=2) - margin, axis=1)
    valid &= np.all(
        base <= np.take_along_axis(parent, index + 1, axis=2) + margin, axis=1
    )
    return np.where(valid, layer, -1)


def _count_not_above(
    rows: np.ndarray, corner: np.ndarray, values: np.ndarray
) -> np.ndarray:
    """Return how many values of a sorted row are not above each given value.

    A binary search on all rows at once.

    Args:
        rows: Sorted rows per column and corner, shape (n, 4, m).
        corner: The corner row to search for each value, shape (n, nlay).
        values: Shape (n, nlay).
    """
    size = rows.shape[2]
    column = np.arange(rows.shape[0])[:, None]
    lower = np.zeros(values.shape, dtype=np.int64)
    upper = np.full(values.shape, size, dtype=np.int64)
    for _ in range(size.bit_length() + 1):
        searching = lower < upper
        middle = (lower + upper) // 2
        above = rows[column, corner, np.minimum(middle, size - 1)] > values
        lower = np.where(searching & ~above, middle + 1, lower)
        upper = np.where(searching & above, middle, upper)
    return lower


def _require_fill(
    layers: np.ndarray,
    child: np.ndarray,
    parent: np.ndarray,
    column_key: np.ndarray,
    tolerance: np.ndarray,
) -> np.ndarray:
    """Keep parent layers only where the child cells fill the parent cell.

    In every child column of the parent column, the child cells of one parent must
    fill it from top to base, with only zero-thickness layers between them, such as
    the padding that :func:`xtgeo.grid_merge` adds. ``column_key`` identifies the
    parent column of each child column, sorted.
    """
    column, child_k = np.nonzero(layers >= 0)
    if column.size == 0:
        return layers
    k = layers[column, child_k]
    order = np.argsort(column * parent.shape[2] + k, kind="stable")
    column, child_k, k = column[order], child_k[order], k[order]
    starts = np.flatnonzero(np.r_[True, (np.diff(column) != 0) | (np.diff(k) != 0)])
    counts = np.diff(np.r_[starts, column.size])
    first, last = child_k[starts], child_k[starts + counts - 1]
    group_column, group_k = column[starts], k[starts]
    top_margin = tolerance[group_column, first][:, None]
    base_margin = tolerance[group_column, last][:, None]
    flat = np.all(np.diff(child, axis=-1) == 0, axis=1)
    flat_before = np.pad(np.cumsum(flat, axis=1), ((0, 0), (1, 0)))
    skipped = flat_before[group_column, last + 1] - flat_before[group_column, first]
    filled = (last - first + 1 == counts + skipped) & np.all(
        np.abs(child[group_column, :, first] - parent[group_column, :, group_k])
        <= top_margin,
        axis=-1,
    )
    filled &= np.all(
        np.abs(child[group_column, :, last + 1] - parent[group_column, :, group_k + 1])
        <= base_margin,
        axis=-1,
    )

    unique_columns, columns_per_parent = np.unique(column_key, return_counts=True)
    group_key = column_key[group_column] * parent.shape[2] + group_k
    unique_groups, inverse, groups = np.unique(
        group_key, return_inverse=True, return_counts=True
    )
    complete = np.bincount(inverse, weights=filled) == groups
    expected = columns_per_parent[
        np.searchsorted(unique_columns, unique_groups // parent.shape[2])
    ]
    complete &= groups == expected

    keep = np.repeat(complete[inverse], counts)
    result = np.full_like(layers, -1)
    result[column[keep], child_k[keep]] = k[keep]
    return result


def _cell_types(
    actnumsv: np.ndarray,
    zcornsv: np.ndarray,
    coarse: NestedHybridBlock,
    parents: np.ndarray,
) -> np.ndarray:
    """Return the :class:`NestedHybridCellType` value of each cell."""
    active = actnumsv > 0
    celltype = np.full(actnumsv.shape, NestedHybridCellType.OTHER_INACTIVE, np.int8)
    # The refined blocks come after the coarse block.
    window, rest = slice(coarse.istart, coarse.istop), slice(coarse.istop, None)
    celltype[window][active[window]] = NestedHybridCellType.ACTIVE_COARSE
    celltype[rest][active[rest]] = NestedHybridCellType.ACTIVE_REFINED

    # Only active refined cells count as children.
    parent_i, parent_j, parent_k = parents[:, rest]
    with_layer = active[rest] & (parent_k >= 0)
    # True for refined columns with an active cell that has no parent layer.
    without_layer = (active[rest] & (parent_i >= 0) & (parent_k < 0)).any(axis=2)

    # Without a parent layer, each cell with thickness in the parent column can
    # be the parent. All layers of a refined column have the same parent column.
    nrow = actnumsv.shape[1]
    column = np.unique(
        parent_i[..., 0][without_layer].astype(np.int64) * nrow
        + parent_j[..., 0][without_layer]
    )
    column_i, column_j = column // nrow, column % nrow
    bounds = _column_bounds(zcornsv, column_i, column_j)
    thick = np.diff(bounds, axis=-1).max(axis=1) > 0
    in_column = celltype[column_i, column_j]
    in_column[thick & (in_column == NestedHybridCellType.OTHER_INACTIVE)] = (
        NestedHybridCellType.UNVERIFIED_PARENT
    )
    celltype[column_i, column_j] = in_column

    index = (parent_i[with_layer], parent_j[with_layer], parent_k[with_layer])
    replaced = ~active[index]
    celltype[tuple(axis[replaced] for axis in index)] = (
        NestedHybridCellType.REPLACED_PARENT
    )
    return celltype
