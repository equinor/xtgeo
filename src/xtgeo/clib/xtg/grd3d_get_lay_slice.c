/*
****************************************************************************************
 *
 * NAME:
 *    grd3d_get_lay_slice.c
 *
 * DESCRIPTION:
 *    Find the corners of the cells and return an array with 5 XY per cell:
 *    (x0, y0, x1, y1, x3, y3, x2, y2, x0, y0,  ....) + an array with ib number
 *    So if 3000 active cells, then the array will be 3000*10 = 30000 entries.
 *    This will e.g. be a basis for matplotlib plotting of layers.
 *
 *    Note that the result in in C order (row fastest)
 *
 * ARGUMENTS:
 *    nx...nz          i     Grid dimensions
 *    p_*              i     Grid geometries arrays with numpy lengths
 *    kslice           i     Requested K slice, start with 1
 *    koption          i     0 for front/upper, 1 for back/lower
 *    actonly          i     If 1 only return active cells
 *    slicev          i/o    Return array with 5 corner XY per cell
 *    nslicev          i     Allocated length of slicev
 *    icv             i/o    Returned ib number of cells
 *    nibv             i     Allocated length of slicev
 *
 * RETURNS:
 *    Number of resulting cells
 *    Updated vectors
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ***************************************************************************************
 */

#include "logger.h"
#include "libxtg.h"
#include "libxtg_.h"


int grd3d_get_lay_slice(
    int nx,
    int ny,
    int nz,

    double *coordsv,
    long ncoordin,
    double *zcornsv,
    long nzcornin,
    int *p_actnum_v,
    long nactin,

    int kslice,
    int koption,
    int actonly,

    double *slicev,
    long nslicev,
    long *icv,
    long nicv
    )

{
    double crs[24];
    int i, j, kshift;
    long ib, ic, icx, icn;

    logger_info(LI, FI, FU, "Getting layer slice: %s", FU);
    logger_info(LI, FI, FU, "Dimens for arrays %ld %ld", nslicev, nicv);

    if (kslice > nz || kslice < 1) {
        logger_warn(LI, FI, FU, "Slice is outside range, return");
        return -1;
    }

    kshift = 0;
    if (koption == 1) kshift = 12;  /* lower cell layer, not upper */

    icx = 0;
    icn = 0;
    for (i = 1; i <= nx; i++) {
        for (j = 1; j <= ny; j++) {
            ib = x_ijk2ib(i, j, kslice, nx, ny, nz, 0);
            ic = x_ijk2ic(i, j, kslice, nx, ny, nz, 0);
            grd3d_corners(i, j, kslice, nx, ny, nz, coordsv, 0,
                          zcornsv, 0, crs);

            if (actonly == 1 && p_actnum_v[ib] == 0) continue;

            slicev[icx++] = crs[0 + kshift]; slicev[icx++] = crs[1 + kshift];
            slicev[icx++] = crs[3 + kshift]; slicev[icx++] = crs[4 + kshift];
            slicev[icx++] = crs[9 + kshift]; slicev[icx++] = crs[10 + kshift];
            slicev[icx++] = crs[6 + kshift]; slicev[icx++] = crs[7 + kshift];
            slicev[icx++] = crs[0 + kshift]; slicev[icx++] = crs[1 + kshift]; /*close*/

            icv[icn++] = ic;
        }

    }

    logger_info(LI, FI, FU, "Getting layer slice done! %s", FU);

    return icn;
}
