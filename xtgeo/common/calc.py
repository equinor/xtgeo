"""
Some common XTGEO calculation routines
"""

import cxtgeo.cxtgeo as _cxtgeo

from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

def ib_to_ijk(ib, nx, ny, nz, ibbase=0):
    """
    Convert a 1D index (starting from ibbase) to cell indices I J K.

    Returns I J K as a tuple.
    """

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

    if ibbase == 0 and ib > nx * ny* nz - 1:
        xtg.warn('Something is wrong with IJK conversion')
        xtg.warn('I J K, NX, NY, NZ IB: {} {} {} {} {} {} {}'
                 .format(i, j, k, nx, ny, nz, ib))

    return ib


def vectorinfo2(x1, x2, y1, y2, option=1):
    """
    Get length and angles from 2 points in space (2D plane).

    Option = 1 gives normal school angle (counterclock from X)
    """

    _cxtgeo.xtg_verbose_file("NONE")

    dbg = xtg.get_syslevel()

    lenp = _cxtgeo.new_doublepointer()
    radp = _cxtgeo.new_doublepointer()
    degp = _cxtgeo.new_doublepointer()

    _cxtgeo.x_vector_info2(x1, x2, y1, y2, lenp, radp, degp, option, dbg)

    llen = _cxtgeo.doublepointer_value(lenp)
    rad = _cxtgeo.doublepointer_value(radp)
    deg = _cxtgeo.doublepointer_value(degp)

    return llen, rad, deg
