"""Private module for merging two separated Grid instances."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from xtgeo.common.log import null_logger

if TYPE_CHECKING:
    from xtgeo.grid3d import Grid

logger = null_logger(__name__)


def merge_grids(
    grid1: Grid,
    grid2: Grid,
) -> Grid:
    """Merge two areally-separated grids into a single grid instance.

    See description in public function (in grid.py)
    """
    from xtgeo.grid3d import Grid

    grid1._set_xtgformat2()
    grid2._set_xtgformat2()

    # Ensure both grids have the same ijk_handedness
    if (
        grid1.ijk_handedness is not None
        and grid2.ijk_handedness is not None
        and grid1.ijk_handedness != grid2.ijk_handedness
    ):
        logger.debug(
            "Grid handedness mismatch: grid1=%s, grid2=%s. "
            "Creating a copy of grid2 and adjusting to match grid1.",
            grid1.ijk_handedness,
            grid2.ijk_handedness,
        )
        grid2 = grid2.copy()
        grid2.ijk_handedness = grid1.ijk_handedness
    elif grid1.ijk_handedness is None or grid2.ijk_handedness is None:
        logger.debug(
            "Cannot verify handedness compatibility: grid1=%s, grid2=%s",
            grid1.ijk_handedness,
            grid2.ijk_handedness,
        )

    new_nlay = max(grid1.nlay, grid2.nlay)

    # Place grid1 at (0, 0) and grid2 with a 1-cell gap
    offset1 = (0, 0)
    offset2 = (grid1.ncol + 1, 0)

    i1_start, j1_start = offset1
    i2_start, j2_start = offset2

    j1_end = j1_start + grid1.nrow
    i2_end = i2_start + grid2.ncol
    j2_end = j2_start + grid2.nrow

    new_ncol = i2_end
    new_nrow = max(j1_end, j2_end)

    logger.debug("Merging grids: %s and %s", grid1.dimensions, grid2.dimensions)
    logger.debug("Result grid dimensions: (%s, %s, %s)", new_ncol, new_nrow, new_nlay)

    new_coordsv = np.zeros((new_ncol + 1, new_nrow + 1, 6), dtype=np.float64)
    new_zcornsv = np.zeros(
        (new_ncol + 1, new_nrow + 1, new_nlay + 1, 4), dtype=np.float32
    )
    new_actnumsv = np.zeros((new_ncol, new_nrow, new_nlay), dtype=np.int32)

    _copy_grid_data(
        grid1,
        new_coordsv,
        new_zcornsv,
        new_actnumsv,
        i1_start,
        j1_start,
        new_nlay,
    )

    _copy_grid_data(
        grid2,
        new_coordsv,
        new_zcornsv,
        new_actnumsv,
        i2_start,
        j2_start,
        new_nlay,
    )

    _fill_gap_pillars(new_coordsv, new_zcornsv, new_actnumsv)

    merged_grid = Grid(
        coordsv=new_coordsv,
        zcornsv=new_zcornsv,
        actnumsv=new_actnumsv,
    )

    # Fix any remaining zero pillars in the merged grid
    merged_grid._get_grid_cpp().fix_zero_pillars()

    _merge_properties(
        merged_grid,
        grid1,
        grid2,
        i1_start,
        j1_start,
        i2_start,
        j2_start,
        new_nlay,
    )

    return merged_grid


def _copy_grid_data(
    source_grid: Grid,
    target_coordsv: np.ndarray,
    target_zcornsv: np.ndarray,
    target_actnumsv: np.ndarray,
    i_offset: int,
    j_offset: int,
    target_nlay: int,
) -> None:
    """Copy grid data from source to target arrays at given offset.

    If source grid has fewer layers than target, the bottom layer geometry
    is extended vertically with inactive cells for the additional layers.

    Args:
        source_grid: Source grid to copy from
        target_coordsv: Target coordinate array to copy into
        target_zcornsv: Target zcorn array to copy into
        target_actnumsv: Target actnum array to copy into
        i_offset: Column offset in target grid
        j_offset: Row offset in target grid
        target_nlay: Number of layers in target grid
    """
    ncol, nrow, nlay = source_grid.dimensions

    # Copy pillar coordinates (shape: ncol+1, nrow+1, 6)
    target_coordsv[
        i_offset : i_offset + ncol + 1,
        j_offset : j_offset + nrow + 1,
        :,
    ] = source_grid._coordsv

    if nlay == target_nlay:
        target_zcornsv[
            i_offset : i_offset + ncol + 1,
            j_offset : j_offset + nrow + 1,
            : nlay + 1,
            :,
        ] = source_grid._zcornsv

        target_actnumsv[
            i_offset : i_offset + ncol,
            j_offset : j_offset + nrow,
            :nlay,
        ] = source_grid._actnumsv
    else:
        # Grid has fewer layers than target - extend bottom layers
        target_zcornsv[
            i_offset : i_offset + ncol + 1,
            j_offset : j_offset + nrow + 1,
            : nlay + 1,
            :,
        ] = source_grid._zcornsv

        bottom_z = source_grid._zcornsv[:, :, nlay, :]  # Shape: (ncol+1, nrow+1, 4)

        # Calculate layer thickness from the last layer
        second_last_z = source_grid._zcornsv[:, :, nlay - 1, :]
        layer_thickness = bottom_z - second_last_z

        for k in range(nlay + 1, target_nlay + 1):
            extension = layer_thickness * (k - nlay)
            target_zcornsv[
                i_offset : i_offset + ncol + 1,
                j_offset : j_offset + nrow + 1,
                k,
                :,
            ] = bottom_z + extension

        # Copy actnum for existing layers
        target_actnumsv[
            i_offset : i_offset + ncol,
            j_offset : j_offset + nrow,
            :nlay,
        ] = source_grid._actnumsv


def _fill_gap_pillars(
    coordsv: np.ndarray, zcornsv: np.ndarray, actnumsv: np.ndarray
) -> None:
    """Fill in pillar coordinates for gap areas using interpolation.

    Only fills pillars that are adjacent to active cells. Gap pillars completely
    surrounded by inactive cells are left with zero coordinates (handled by C++).

    For pillars that need filling:
    1. Uses bilinear interpolation from surrounding non-zero pillars if available
    2. Otherwise falls back to nearest non-zero pillar
    3. Ensures pillar coordinates form valid straight vertical pillars

    Args:
        coordsv: Coordinate array (ncol+1, nrow+1, 6) to fill
        zcornsv: Z-corner array (ncol+1, nrow+1, nlay+1, 4) to fill
        actnumsv: Active cell array (ncol, nrow, nlay) to check which pillars are needed
    """
    logger.debug("Filling gap pillars adjacent to active cells...")
    ncol_p1, nrow_p1, _ = coordsv.shape

    # Find pillars that are still zero (gap areas)
    pillar_empty = np.all(coordsv == 0.0, axis=2)

    if not np.any(pillar_empty):
        return  # No gaps to fill

    pillars_needed = _find_pillars_needed(pillar_empty, actnumsv)

    n_to_fill = np.sum(pillars_needed)
    if n_to_fill == 0:
        logger.debug("No gap pillars need filling (all surrounded by inactive cells)")
        return

    needed_indices = np.argwhere(pillars_needed)
    for i, j in needed_indices:
        # Try interpolation first
        coords, z_vals = _interpolate_pillar(coordsv, zcornsv, i, j, pillar_empty)

        if coords is None:
            # Fall back to nearest neighbor
            coords, z_vals = _find_nearest_pillar(coordsv, zcornsv, i, j, pillar_empty)

        if coords is not None:
            coordsv[i, j, :] = _ensure_vertical_pillar(coords)

        if z_vals is not None:
            zcornsv[i, j, :, :] = _ensure_z_separation(z_vals)

    logger.debug("Filling gap pillars adjacent to active cells... done!")


def _find_pillars_needed(pillar_empty: np.ndarray, actnumsv: np.ndarray) -> np.ndarray:
    """Identify gap pillars that are adjacent to active cells.

    Args:
        pillar_empty: Boolean array (ncol+1, nrow+1) marking empty pillars
        actnumsv: Active cell array (ncol, nrow, nlay)

    Returns:
        Boolean array (ncol+1, nrow+1) marking which empty pillars need filling
    """
    logger.debug("Find pillars needed...")
    ncol_p1, nrow_p1 = pillar_empty.shape
    pillars_needed = np.zeros_like(pillar_empty, dtype=bool)

    # Vectorized: Find all cells that have any active layer
    has_active = np.any(actnumsv > 0, axis=2)  # (ncol, nrow)

    # For each active cell, mark its 4 corner pillars as needed
    # A cell at (i, j) uses pillars at (i, j), (i+1, j), (i, j+1), (i+1, j+1)
    active_cells = np.argwhere(has_active)
    for ci, cj in active_cells:
        for di, dj in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            pi, pj = ci + di, cj + dj
            if 0 <= pi < ncol_p1 and 0 <= pj < nrow_p1:
                pillars_needed[pi, pj] = True

    logger.debug("Find pillars needed... done!")
    return pillars_needed & pillar_empty


def _ensure_vertical_pillar(coords: np.ndarray) -> np.ndarray:
    """Ensure pillar coordinates define a proper vertical pillar.

    A vertical pillar should have:
    - Same X, Y coordinates at top and bottom (vertical)
    - Different Z coordinates at top and bottom (non-degenerate)

    Args:
        coords: Pillar coordinates [x_top, y_top, z_top, x_bot, y_bot, z_bot]

    Returns:
        Adjusted coordinates ensuring vertical geometry
    """
    result = coords.copy()

    # Check if pillar is all zeros (uninitialized)
    if np.all(result == 0.0):
        # Even for unused pillars, ensure Z coordinates are different
        result[5] = 1e-5  # z_bot > z_top (both 0)
        return result

    # Make pillar vertical: bottom X,Y = top X,Y
    result[3] = result[0]  # x_bot = x_top
    result[4] = result[1]  # y_bot = y_top

    # Ensure top and bottom Z are different (avoid degenerate pillar)
    # Use tolerance > 1e-6 (C++ TOLERANCE) to avoid RMS API warnings
    min_z_separation = 1e-5
    if abs(result[5] - result[2]) < min_z_separation:
        # Extend pillar downward by minimum separation
        if result[2] != 0.0:  # Non-zero reference Z
            result[5] = result[2] + min_z_separation
        else:
            result[5] = min_z_separation

    return result


def _ensure_z_separation(z_vals: np.ndarray) -> np.ndarray:
    """Ensure Z corner values have proper vertical separation.

    Checks each layer interface and ensures adjacent layers have minimum separation.

    Args:
        z_vals: Z corner array (nlay+1, 4) for all layers at 4 corners

    Returns:
        Adjusted Z values with proper separation
    """
    result = z_vals.copy()
    nlay_p1 = result.shape[0]

    # Check if all zeros (uninitialized pillar)
    if np.all(result == 0.0):
        # Leave it as zeros - it's not needed
        return result

    min_separation = 1e-5  # Minimum vertical separation in same units as Z

    # For each layer interface, ensure proper separation
    for k in range(nlay_p1 - 1):
        for corner in range(4):
            z_current = result[k, corner]
            z_next = result[k + 1, corner]

            if abs(z_next - z_current) < min_separation:
                # Adjust next layer to maintain proper separation
                if k == 0:
                    result[k + 1, corner] = z_current + min_separation
                else:
                    prev_thickness = result[k, corner] - result[k - 1, corner]
                    if abs(prev_thickness) > min_separation:
                        result[k + 1, corner] = z_current + prev_thickness
                    else:
                        result[k + 1, corner] = z_current + min_separation

    return result


def _interpolate_pillar(
    coordsv: np.ndarray,
    zcornsv: np.ndarray,
    i: int,
    j: int,
    pillar_empty: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Interpolate pillar coordinates from surrounding non-empty pillars.

    Searches for the nearest non-empty pillar in each of the 4 quadrants around
    the empty pillar position. If at least 2 pillars are found, returns the
    simple average of all found pillars' coordinates and Z-values.

    Args:
        coordsv: Coordinate array
        zcornsv: Z-corner array
        i: Column index of empty pillar
        j: Row index of empty pillar
        pillar_empty: Boolean array marking empty pillars

    Returns:
        Tuple of (averaged_coords, averaged_z) or (None, None) if fewer than
        2 neighbors found
    """
    ncol_p1, nrow_p1 = pillar_empty.shape

    # Try to find 4 corners for interpolation
    # Look for nearest non-empty pillar in each quadrant
    corners = {}
    for di_sign, dj_sign in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        for radius in range(1, max(ncol_p1, nrow_p1)):
            ni, nj = i + radius * di_sign, j + radius * dj_sign
            if 0 <= ni < ncol_p1 and 0 <= nj < nrow_p1 and not pillar_empty[ni, nj]:
                corners[(di_sign, dj_sign)] = (ni, nj)
                break

    # If we have at least 2 opposite corners, we can interpolate
    if len(corners) >= 2:
        # Simple averaging of available corners
        coords_list = []
        z_list = []
        for ni, nj in corners.values():
            coords_list.append(coordsv[ni, nj, :])
            z_list.append(zcornsv[ni, nj, :, :])

        avg_coords = np.mean(coords_list, axis=0)
        avg_z = np.mean(z_list, axis=0)

        return avg_coords, avg_z

    return (None, None)


def _find_nearest_pillar(
    coordsv: np.ndarray,
    zcornsv: np.ndarray,
    i: int,
    j: int,
    pillar_empty: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Find the nearest non-empty pillar to position (i, j).

    Args:
        coordsv: Coordinate array
        zcornsv: Z-corner array
        i: Column index of empty pillar
        j: Row index of empty pillar
        pillar_empty: Boolean array marking empty pillars

    Returns:
        Tuple of (nearest_coords, nearest_z) or (None, None) if no neighbor found
    """
    ncol_p1, nrow_p1 = pillar_empty.shape

    # Search in expanding squares around (i, j)
    for radius in range(1, max(ncol_p1, nrow_p1)):
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                # Only check perimeter of current square
                if abs(di) != radius and abs(dj) != radius:
                    continue

                ni, nj = i + di, j + dj

                if 0 <= ni < ncol_p1 and 0 <= nj < nrow_p1 and not pillar_empty[ni, nj]:
                    return coordsv[ni, nj, :].copy(), zcornsv[ni, nj, :, :].copy()

    return (None, None)


def _merge_properties(
    merged_grid: Grid,
    grid1: Grid,
    grid2: Grid,
    i1_start: int,
    j1_start: int,
    i2_start: int,
    j2_start: int,
    target_nlay: int,
) -> None:
    """Merge properties from both input grids into the merged grid.

    For properties with the same name:
    - Continuous properties: Merge values
    - Discrete properties: Merge if codes match, otherwise rename grid2's property

    Args:
        merged_grid: The output merged grid
        grid1: First input grid
        grid2: Second input grid
        i1_start: Column offset for grid1
        j1_start: Row offset for grid1
        i2_start: Column offset for grid2
        j2_start: Row offset for grid2
        target_nlay: Number of layers in merged grid
    """

    if not grid1.props and not grid2.props:
        return

    # Build a dict of properties to merge
    prop_map: dict[str, list[tuple[Grid, int, int, str]]] = {}

    # Add grid1 properties
    if grid1.props:
        for prop in grid1.props:
            if prop.name is not None:
                prop_map.setdefault(prop.name, []).append(
                    (grid1, i1_start, j1_start, prop.name)
                )

    # Add grid2 properties
    if grid2.props:
        for prop in grid2.props:
            if prop.name is None:
                continue
            # Check if property name already exists in grid1
            g1_prop = grid1.get_prop_by_name(prop.name) if grid1.props else None
            if g1_prop is not None:
                g2_prop = prop

                # Decide if we can merge or need to rename
                if g1_prop.isdiscrete == g2_prop.isdiscrete:
                    if g1_prop.isdiscrete:
                        # Both discrete - check if codes match
                        if g1_prop.codes == g2_prop.codes:
                            # Same codes, can merge
                            prop_map[prop.name].append(
                                (grid2, i2_start, j2_start, prop.name)
                            )
                        else:
                            # Different codes, rename grid2's property
                            new_name = _find_unique_name(prop.name, prop_map)
                            prop_map[new_name] = [
                                (grid2, i2_start, j2_start, prop.name)
                            ]
                            logger.debug(
                                "Property '%s' from grid2 has different "
                                "codes than grid1, renaming to '%s'",
                                prop.name,
                                new_name,
                            )
                    else:
                        # Both continuous, merge them
                        prop_map[prop.name].append(
                            (grid2, i2_start, j2_start, prop.name)
                        )
                else:
                    # One discrete, one continuous - rename grid2's property
                    new_name = _find_unique_name(prop.name, prop_map)
                    prop_map[new_name] = [(grid2, i2_start, j2_start, prop.name)]
                    logger.debug(
                        "Property '%s' from grid2 has different type "
                        "than grid1, renaming to '%s'",
                        prop.name,
                        new_name,
                    )
            else:
                # New property, add it
                prop_map[prop.name] = [(grid2, i2_start, j2_start, prop.name)]

    for prop_name, prop_sources in prop_map.items():
        _create_merged_property(merged_grid, prop_name, prop_sources, target_nlay)


def _find_unique_name(base_name: str, existing_names: dict) -> str:
    """Find a unique property name by appending a suffix.

    Args:
        base_name: Base name to make unique
        existing_names: Dict of existing property names

    Returns:
        Unique property name
    """
    if base_name not in existing_names:
        return base_name

    counter = 2
    while f"{base_name}_{counter}" in existing_names:
        counter += 1

    return f"{base_name}_{counter}"


def _create_merged_property(
    merged_grid: Grid,
    prop_name: str,
    sources: list[tuple[Grid, int, int, str]],
    target_nlay: int,
) -> None:
    """Create a merged property from multiple source grids.

    Args:
        merged_grid: The output merged grid
        prop_name: Name of the property in the merged grid
        sources: List of (source_grid, offset_i, offset_j, source_prop_name) tuples
        target_nlay: Number of layers in merged grid
    """
    from xtgeo.grid3d import GridProperty

    # Get the first source property to determine type and initialize
    first_grid, _, _, first_prop_name = sources[0]
    first_prop = first_grid.get_prop_by_name(first_prop_name)
    if first_prop is None:
        raise ValueError(f"Property '{first_prop_name}' not found in first grid")

    # Initialize merged property array with zeros (or appropriate default)
    merged_values = np.zeros(
        (merged_grid.ncol, merged_grid.nrow, merged_grid.nlay),
        dtype=first_prop.values.dtype,
    )

    # Copy values from each source grid
    for source_grid, offset_i, offset_j, source_prop_name in sources:
        source_prop = source_grid.get_prop_by_name(source_prop_name)
        if source_prop is None:
            raise ValueError(f"Property '{source_prop_name}' not found in source grid")
        ncol, nrow, nlay = source_grid.dimensions

        if nlay == target_nlay:
            merged_values[
                offset_i : offset_i + ncol,
                offset_j : offset_j + nrow,
                :nlay,
            ] = source_prop.values
        else:
            merged_values[
                offset_i : offset_i + ncol,
                offset_j : offset_j + nrow,
                :nlay,
            ] = source_prop.values

    GridProperty(
        merged_grid,
        name=prop_name,
        discrete=first_prop.isdiscrete,
        values=merged_values,
        codes=first_prop.codes if first_prop.isdiscrete else None,
    )
