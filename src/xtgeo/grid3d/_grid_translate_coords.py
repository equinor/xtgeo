"""Private module for translating coordiantes plus flipping and rotation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from xtgeo.grid3d import Grid


def _rotate_grid3d(
    new_grd: Grid, add_rotation: float, rotation_xy: tuple[float, float] | None = None
) -> None:
    """Rotate the grid around the cell 1,1,1 corner, or some other coordinate."""

    # Convert angle to radians
    angle_rad = math.radians(add_rotation)

    # Create rotation matrix coefficients
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)

    # extract
    coord_array = new_grd._coordsv.copy()
    rotated_coords = coord_array.copy()

    if rotation_xy is None:
        x0, y0, _ = new_grd._coordsv[0, 0, :3]
    else:
        x0, y0 = rotation_xy

    x_coords = coord_array[:, :, [0, 3]].copy()  # x1 and x2
    y_coords = coord_array[:, :, [1, 4]].copy()  # y1 and y2

    # Translate to origin
    x_translated = x_coords - x0
    y_translated = y_coords - y0

    # Rotate using rotation matrix
    x_rotated = x_translated * cos_theta - y_translated * sin_theta
    y_rotated = x_translated * sin_theta + y_translated * cos_theta

    # Translate back and assign
    rotated_coords[:, :, [0, 3]] = x_rotated + x0
    rotated_coords[:, :, [1, 4]] = y_rotated + y0

    # Z coordinates remain unchanged (indices 2 and 5)
    new_grd._coordsv = rotated_coords.copy()


def _flip_vertically(grid: Grid) -> None:
    """Flip the grid vertically."""

    # find average depth of corners
    avg_z = grid._zcornsv.mean()

    grid._zcornsv = grid._zcornsv[:, :, ::-1, :]
    grid._zcornsv *= -1

    # find the new average and compute the difference for shifting
    new_avg_z = grid._zcornsv.mean()
    diff = avg_z - new_avg_z
    grid._zcornsv += diff

    grid._actnumsv = np.flip(grid._actnumsv, axis=2).copy()

    # Handle properties if they exist
    if grid._props and grid._props.props:
        for prop in grid._props.props:
            prop.values = np.flip(prop.values, axis=2).copy()

    # When we flip the grid, the subgrid info must also be flipped
    subgrids = grid.get_subgrids()
    if subgrids:
        reverted = dict(reversed(subgrids.items()))
        grid.set_subgrids(reverted)

    if grid._ijk_handedness == "left":
        grid._ijk_handedness = "right"
    else:
        grid._ijk_handedness = "left"


def _translate_geometry(grid: Grid, translate: tuple[float, float, float]) -> None:
    grid._coordsv[:, :, 0] += translate[0]
    grid._coordsv[:, :, 1] += translate[1]
    grid._coordsv[:, :, 2] += translate[2]
    grid._coordsv[:, :, 3] += translate[0]
    grid._coordsv[:, :, 4] += translate[1]
    grid._coordsv[:, :, 5] += translate[2]

    grid._zcornsv += translate[2]


def _transfer_to_target(grid: Grid, target: tuple[float, float, float]) -> None:
    # get the coordinates for the active cells
    x, y, z = grid.get_xyz(asmasked=True)

    # set the grid center to the desired target coordinates
    shift_x = target[0] - x.values.mean()
    shift_y = target[1] - y.values.mean()
    shift_z = target[2] - z.values.mean()

    _translate_geometry(grid, (shift_x, shift_y, shift_z))


def translate_coordinates(
    self: Grid,
    translate: tuple[float, float, float] = (0.0, 0.0, 0.0),
    flip: tuple[int, int, int] = (1, 1, 1),
    add_rotation: float = 0.0,
    rotation_point: tuple[float, float] | None = None,
    target_coordinates: tuple[float, float, float] | None = None,
) -> None:
    """Rotate, flip grid and translate grid coordinates.

    This should be done in a sequence like this
    1) Add rotation (with value ``add_rotation``) to rotate the grid counter-clockwise
    2) Flip the grid
    3a) translate the grid by adding (x, y, z), OR
    3b) set a target location the grid centre.

    """
    self._set_xtgformat2()

    if abs(add_rotation) > 1e-10:  # Skip rotation if angle is essentially zero
        _rotate_grid3d(self, add_rotation, rotation_xy=rotation_point)

    if flip[2] == -1:
        _flip_vertically(self)

    if flip[0] == -1:
        self.reverse_column_axis()

    if flip[1] == -1:
        self.reverse_row_axis()

    use_translate = False
    if not all(abs(x) < 1e-10 for x in translate):
        _translate_geometry(self, translate)
        use_translate = True

    if target_coordinates and use_translate:
        raise ValueError(
            "Using both key 'translate' and key 'target_coordinates' is not allowed. "
            "Use either."
        )

    if target_coordinates:
        # transfer the grid's geometrical centre to a given location
        _transfer_to_target(self, target_coordinates)
