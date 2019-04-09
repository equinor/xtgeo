"""Some common XTGEO calculation routines."""

import numpy as np
import xtgeo.cxtgeo.cxtgeo as _cxtgeo

from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)

DBG = xtg.get_syslevel()


def ib_to_ijk(ib, nx, ny, nz, ibbase=0):
    """
    Convert a 1D index (starting from ibbase) to cell indices I J K.

    Returns I J K as a tuple.
    """

    logger.info("IB to IJK")

    ip = _cxtgeo.new_intpointer()
    jp = _cxtgeo.new_intpointer()
    kp = _cxtgeo.new_intpointer()

    _cxtgeo.x_ib2ijk(ib, ip, jp, kp, nx, ny, nz, ibbase)

    i = _cxtgeo.intpointer_value(ip)
    j = _cxtgeo.intpointer_value(jp)
    k = _cxtgeo.intpointer_value(kp)

    return (i, j, k)


def ijk_to_ib(i, j, k, nx, ny, nz, ibbase=0, forder=True):
    """
    Convert a cell indices I J K to 1D index (starting from ibbase).

    Both Fortran order and C order (Fortran order is default).
    """

    if forder:
        # fortran order
        ib = _cxtgeo.x_ijk2ib(i, j, k, nx, ny, nz, ibbase)
    else:
        # c order
        ib = _cxtgeo.x_ijk2ic(i, j, k, nx, ny, nz, ibbase)

    if ibbase == 0 and ib > nx * ny * nz - 1:
        xtg.warn("Something is wrong with IJK conversion")
        xtg.warn(
            "I J K, NX, NY, NZ IB: {} {} {} {} {} {} {}".format(i, j, k, nx, ny, nz, ib)
        )

    return ib


def vectorinfo2(x1, x2, y1, y2, option=1):
    """
    Get length and angles from 2 points in space (2D plane).

    Option = 1 gives normal school angle (counterclock from X)
    """

    _cxtgeo.xtg_verbose_file("NONE")

    lenp = _cxtgeo.new_doublepointer()
    radp = _cxtgeo.new_doublepointer()
    degp = _cxtgeo.new_doublepointer()

    _cxtgeo.x_vector_info2(x1, x2, y1, y2, lenp, radp, degp, option, DBG)

    llen = _cxtgeo.doublepointer_value(lenp)
    rad = _cxtgeo.doublepointer_value(radp)
    deg = _cxtgeo.doublepointer_value(degp)

    return llen, rad, deg


def find_flip(xv, yv, zv):
    """Find the flip status by computing the cross products.

    If flip is 1, then the system is left-handed, typically
    X is East, Y is North and Z downwards.

    Args:
        xv (tuple): First vector (x1, y1, z1)
        yv (tuple): Second vector (x2, y2, z2)
        zv (tuple): Third vector (x3, y3, z3)

    Return:
        Flip flag (1 of -1)

"""
    flip = 0

    xv = np.array(xv)
    yv = np.array(yv)
    zv = np.array(zv)

    xycross = np.cross(xv, yv)

    logger.debug("Cross product XY is %s", xycross)

    if xycross[2] < 0:
        flip = 1
    else:
        flip = -1

    return flip
