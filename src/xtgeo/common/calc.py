"""Some common XTGEO calculation routines."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from xtgeo import XTGeoCLibError, _cxtgeo
from xtgeo.common import XTGeoDialog, _angles, null_logger

xtg = XTGeoDialog()
logger = null_logger(__name__)


def ib_to_ijk(
    ib: int,
    nx: int,
    ny: int,
    nz: int,
    ibbase: int = 0,
    forder: bool = True,
) -> tuple[int, int, int]:
    """Convert a 1D index (starting from ibbase) to cell indices I J K.

    The default is F-order, but ``forder=False`` gives C order

    Returns I J K as a tuple.
    """

    logger.info("IB to IJK")

    if forder:
        iv, jv, kv = _cxtgeo.x_ib2ijk(ib, nx, ny, nz, ibbase)
    else:
        iv, jv, kv = _cxtgeo.x_ic2ijk(ib, nx, ny, nz, ibbase)

    return (iv, jv, kv)


def ijk_to_ib(
    i: int,
    j: int,
    k: int,
    nx: int,
    ny: int,
    nz: int,
    ibbase: int = 0,
    forder: bool = True,
) -> int:
    """
    Convert cell indices I J K to 1D index (starting from ibbase).

    Both Fortran order and C order (Fortran order is default).
    """

    if forder:
        # fortran order
        ib = _cxtgeo.x_ijk2ib(i, j, k, nx, ny, nz, ibbase)
    else:
        # c order
        ib = _cxtgeo.x_ijk2ic(i, j, k, nx, ny, nz, ibbase)
    if ib < 0:
        raise IndexError(f"Negative index: {ib}")
    if ibbase == 0 and ib > nx * ny * nz - 1:
        xtg.warn("Something is wrong with IJK conversion")
        xtg.warn(f"I J K, NX, NY, NZ IB: {i} {j} {k} {nx} {ny} {nz} {ib}")

    return ib


def xyori_from_ij(
    iind: int,
    jind: int,
    xcor: float,
    ycor: float,
    xinc: float,
    yinc: float,
    ncol: int,
    nrow: int,
    yflip: int,
    rotation: float,
) -> tuple[float, float]:
    """Get xori and yori given X Y, geometrics and indices for regular maps/cubes.

    Args:
        iind: I index (zero based)
        jind: J index (zero based)
        xcor: X coordinate
        ycor: Y coordinate
        xinc: X increment (in non-rotated space)
        yinc: Y increment (in non-rotated space)
        ncol: Number of columns
        nrow: Number of rows
        yflip: YFLIP (handedness) indicator, 1 og -1
        rotation: Rotation in degrees, anticlock from X axis

    """

    if iind >= ncol or iind < 0 or jind >= nrow or jind < 0:
        raise ValueError(
            f"Indices out of range, offending indices are I, J = ({iind}, {jind}) "
            f"and valid ranges are (0.. {ncol - 1}, 0.. {nrow - 1})",
        )

    # the C library and indices with base 1; hence ned to add 1
    ier, xori, yori = _cxtgeo.surf_xyori_from_ij(
        iind + 1,
        jind + 1,
        xcor,
        ycor,
        xinc,
        yinc,
        ncol,
        nrow,
        yflip,
        rotation,
        0,
    )
    if ier != 0:
        raise RuntimeError(f"Error code {ier} from _cxtgeo.surf_xyori_from_ij")

    return xori, yori


def vectorinfo2(
    x1: float,
    x2: float,
    y1: float,
    y2: float,
    option: int = 1,
) -> tuple[float, float, float]:
    """
    Get length and angles from 2 points in space (2D plane).

    Option = 1 gives normal school angle (counterclock from X), while 0 gives azimuth:
    positive direction clockwise from North.
    """

    llen, rad, deg = _cxtgeo.x_vector_info2(x1, x2, y1, y2, option)

    return llen, rad, deg


def diffangle(angle1: float, angle2: float, option: int = 1) -> float:
    """
    Find the minimim difference between two angles, option=1 means degrees,
    otherwise radians. The routine think clockwise for differences.

    Examples::

        res = diffangle(30, 40)  # res shall be -10
        res = diffangle(360, 170)  # res shall be -170
    """

    return _cxtgeo.x_diff_angle(angle1, angle2, option)


def averageangle(anglelist: Sequence[float]) -> float:
    """
    Find the average of a list of angles, in degrees
    """
    return _cxtgeo.x_avg_angles(anglelist)


def find_flip(xv: tuple[float], yv: tuple[float]) -> Literal[-1, 1]:
    """Find the XY flip status by computing the cross products.

    If flip is 1, then the system is right handed in school algebra
    but left-handed in reservoir models (where typically
    X is East, Y is North, assuming Z downwards).

    Args:
        xv (tuple): First vector (x1, y1, z1)
        yv (tuple): Second vector (x2, y2, z2)

    Return:
        Flip flag (1 or -1)

    .. versionchanged:: 2.1 Reverse the number returned, skip zv
    """

    xv = np.array(xv)
    yv = np.array(yv)

    xycross = np.cross(xv, yv)

    logger.debug("Cross product XY is %s", xycross)

    return -1 if xycross[2] < 0 else 1


def angle2azimuth(
    inangle: float,
    mode: Literal["degrees", "radians"] = "degrees",
) -> float:
    """Return the Azimuth angle given input normal angle.

    Normal angle means counterclock rotation from X (East) axis, while
    azimuth is clockwise rotation from Y (North)

    Args:
        inangle (float): Input angle in normal manner ("school")
        mode (str): "degrees" (default) or "radians"

    Return:
        Azimuth angle (in degrees or radian)
    """
    return (
        _angles._deg_angle2azimuth(inangle)
        if mode == "degrees"
        else _angles._rad_angle2azimuth(inangle)
    )


def azimuth2angle(
    inangle: float,
    mode: Literal["degrees", "radians"] = "degrees",
) -> float:
    """Return the "school" angle given input azimuth angle.

    Normal "school" angle means counterclock rotation from X (East) axis, while
    azimuth is clockwise rotation from Y (North)

    Args:
        inangle (float): Input angle in azimuth manner
        mode (str): "degrees" (default) or "radians"

    Return:
        Angle (in degrees or radians)
    """
    return (
        _angles._deg_azimuth2angle(inangle)
        if mode == "degrees"
        else _angles._rad_azimuth2angle(inangle)
    )


def tetrehedron_volume(vertices: Sequence[float]) -> float:
    """Compute volume of an irregular tetrahedron

    Input is an array of lenght 12 element, and is "list-like" meaning that
    both lists and contiguous numpy arrays are accepted

    Args:
        vertices (list-like): Vertices as e.g. numpy array [[x1, y1, z1], [x2, y2, ...]

    Returns:
        Volume
    """

    vertices = np.array(vertices, dtype=np.float64)

    return _cxtgeo.x_tetrahedron_volume(vertices)


def point_in_tetrahedron(
    x0: float,
    y0: float,
    z0: float,
    vertices: Sequence[float],
) -> bool:
    """Check if point P0 is inside a tetrahedron.

    Args:
        x0 (double): X xoord of point P0
        y0 (double): Y xoord of point P0
        z0 (double): Z xoord of point P0
        vertices (list-like): Vertices as e.g. numpy array [[x1, y1, z1], [x2, y2, ...]

    Returns:
        True of inside or on edge, False else
    """

    vertices = np.array(vertices, dtype=np.float64)

    status = _cxtgeo.x_point_in_tetrahedron(x0, y0, z0, vertices)
    if status == 1:
        raise XTGeoCLibError("Error in x_point_in_tetrahedron")
    if status == 100:
        return True

    return False


def point_in_hexahedron(
    x0: float,
    y0: float,
    z0: float,
    vertices: Sequence[float],
    _algorithm: int = 1,
) -> bool:
    """Check if point P0 is inside a tetrahedron.

    Vertices my be in order of what 3D cells normally have

    ::

       3        4     7        8      Note in C code, cell corners may be starting
       |--------|     |--------|      with 0 index, not 1 as shown here
       |  top   |     |        |
       |        |     |        |
       |--------|     |--------|
       1        2     5        6


    Args:
        x0 (float): X xoord of point P0
        y0 (float): Y xoord of point P0
        z0 (float): Z xoord of point P0
        vertices (list-like): Vertices as e.g. numpy array [[x1, y1, z1], [x2, y2, ...]
        _algorithm (int): Method for calculation (experimental, default may change)

    Returns:
        True of inside or on edge, False else
    """

    vertices = np.array(vertices, dtype=np.float64)

    status = _cxtgeo.x_point_in_hexahedron(x0, y0, z0, vertices, _algorithm)

    return True if status >= 1 else False


def vectorpair_angle3d(
    p0: Sequence[float],
    p1: Sequence[float],
    p2: Sequence[float],
    degrees: bool = True,
    birdview: bool = False,
) -> float | None:
    """Find angle in 3D between two vectors

    ::

             / p1
            /
           / ) a
          /---------------- p2
        p0

    Args:
        p0 (list like): Common point P0 (x y z)
        p1 (list like): Point P1 (x y z)
        p2 (list like): Point P2 (x y z)
        degrees (bool): The result in degrees if True, radians if False
        birdview (bool): If True, find angles projected in Z (bird perspective)

    Returns:
        Angle. If some problem, e.g. one vector is too short, None is returned

    Raises:
        ValueError: Errors in input dimensions, all points must have 3 values
    """

    _p0 = np.array(p0)
    _p1 = np.array(p1)
    _p2 = np.array(p2)

    if any(size != 3 for size in (_p0.size, _p1.size, _p2.size)):
        raise ValueError("Errors in input dimensions, all points must have 3 values")

    degs = 1 if degrees else 0
    bird = 1 if birdview else 0

    angle = _cxtgeo.x_vectorpair_angle3d(_p0, _p1, _p2, degs, bird)

    return None if np.isclose(angle, -999) else angle


def _swap_axes(
    rotation: float,
    yflip: int,
    **values: dict[str, Any],
) -> tuple[float, int, dict[str, np.ndarray]]:
    swapped_values: dict[str, np.ndarray] = {
        name: np.ascontiguousarray(np.swapaxes(val, 0, 1))
        for name, val in values.items()
    }

    # TODO: use a while loop or similar, to compensate for possibly
    # multiple rotations in either direction (yflip is not checked)
    # (or use modulo  (%) on yflip)
    rotation = rotation + yflip * 90
    if rotation < 0:
        rotation += 360
    if rotation >= 360:
        rotation -= 360

    return rotation, yflip * -1, swapped_values
