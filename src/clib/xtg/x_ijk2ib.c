/*
 ***************************************************************************************
 *
 * NAME:
 *   x_ijk2ib.c
 *
 * DESCRIPTION:
 *    Convert (i, j, k) coordinates to a 1-dimensional index
 *
 *    x_ijk2ib()    Fortran order
 *    x_ijk2ic()    C order
 *
 * ARGUMENTS:
 *     i, j, k      cell indices
 *     nx, ny, nz   grid dimensions
 *     ia_start     1-dimensional array index to start from
 *
 * RETURNS:
 *    The (i, j, k) cell's 1-dimensional array index
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ***************************************************************************************
 */

#include "libxtg_.h"

/* Fortran order counting (column major order: i loops fastest, then j, then k) */
long
x_ijk2ib(long i, long j, long k, long nx, long ny, long nz, int ia_start)
{

    if (i > nx || j > ny || k > nz) {
        return -2;
    } else if (i < 1 || j < 1 || k < 1) {
        return -2;
    }

    long ib = (k - 1) * nx * ny;
    ib = ib + (j - 1) * nx;
    ib = ib + i;

    if (ia_start == 0)
        ib--;

    return ib;
}

/* C order counting (row major order: k loops fastest, then j, then i) */
long
x_ijk2ic(long i, long j, long k, long nx, long ny, long nz, int ia_start)
{

    if (i > nx || j > ny || k > nz) {
        return -2;
    } else if (i < 1 || j < 1 || k < 1) {
        return -2;
    }

    long ic = (i - 1) * nz * ny;
    ic = ic + (j - 1) * nz;
    ic = ic + k;

    if (ia_start == 0)
        ic--;

    return ic;
}
