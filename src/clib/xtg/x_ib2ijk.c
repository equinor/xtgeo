
/*
 ***************************************************************************************
 * Change for sequence counting to I,J,K, number. The I,J,K is always
 * base 1 offset. The IB may be base 0 (most common) or base 1, and that
 * is given by the value of ia_start.
 ***************************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"
#include "logger.h"

void
x_ib2ijk(long ib, int *i, int *j, int *k, int nx, int ny, int nz, int ia_start)
{
    long ir, nxy;
    long ix = 1, iy = 1, iz = 1;

    nxy = nx * ny;

    if (ia_start == 0)
        ib = ib + 1; /* offset number to counter number */

    iz = ib / nxy;
    if (iz * nxy < ib)
        iz = iz + 1;
    ir = ib - ((iz - 1) * nxy);
    iy = ir / nx;
    if (iy * nx < ir)
        iy = iy + 1;

    ix = ir - ((iy - 1) * nx);

    if (ix < 1 || ix > nx || iy < 1 || iy > ny || iz < 1 || iz > nz) {
        ix = -99;
        iy = -99;
        iz = -99;
        logger_critical(LI, FI, FU, "Critical error (bug) from %s", FU);
    }

    /* values to return */
    *i = ix;
    *j = iy;
    *k = iz;
}

/* C order: */
void
x_ic2ijk(long ic, int *i, int *j, int *k, int nx, int ny, int nz, int ia_start)
{
    long ir, nzy;
    long ix = 1, iy = 1, iz = 1;

    nzy = nz * ny;

    if (ia_start == 0)
        ic = ic + 1; /* offset number to counter number */

    ix = ic / nzy;
    if (ix * nzy < ic)
        ix = ix + 1;
    ir = ic - ((ix - 1) * nzy);
    iy = ir / nz;
    if (iy * nz < ir)
        iy = iy + 1;

    iz = ir - ((iy - 1) * nz);

    if (ix < 1 || ix > nx || iy < 1 || iy > ny || iz < 1 || iz > nz) {
        ix = -99;
        iy = -99;
        iz = -99;
        logger_critical(LI, FI, FU, "Critical error (bug) from %s", FU);
    }

    /* values to return */
    *i = ix;
    *j = iy;
    *k = iz;
}
