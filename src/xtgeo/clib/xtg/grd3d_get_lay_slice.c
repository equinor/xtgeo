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
 * ARGUMENTS:
 *    nx...nz          i     Grid dimensions
 *    p_*              i     Grid geometries arrays
 *    kslice           i     Requested K slice, start with 1
 *    koption          i     0 for front/upper, 1 for back/lower
 *    actonly          i     If 1 only return active cells
 *    slicev          i/o    Return array with 5 corner XY per cell
 *    nslicev          i     Allocated length of slicev
 *    ibv             i/o    Returned ib number of cells
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
    double *p_coord_v,
    double *p_zcorn_v,
    int *p_actnum_v,
    int kslice,
    int koption,
    int actonly,

    double *slicev,
    long nslicev,
    long *ibv,
    long nibv
    )

{
    double crs[24];
    int i, j, kshift;
    long ib, ic, ibn;

    logger_init(__FILE__, __FUNCTION__);
    logger_info(__LINE__, "Getting layer slice: %s", __FUNCTION__);

    if (kslice > nz || kslice < 1) {
        logger_warn(__LINE__, "Slice is outside range, return");
        return -1;
    }

    kshift = 0;
    if (koption == 1) kshift = 12;  /* lower cell layer, not upper */

    ic = 0;
    ibn = 0;
    for (j = 1; j <= ny; j++) {
        for (i = 1; i <= nx; i++) {
            ib = x_ijk2ib(i, j, kslice, nx, ny, nz, 0);
            grd3d_corners(i, j, kslice, nx, ny, nz, p_coord_v,
                          p_zcorn_v, crs, 0);

            if (actonly == 1 && p_actnum_v[ib] == 0) continue;

            slicev[ic++] = crs[0 + kshift]; slicev[ic++] = crs[1 + kshift];
            slicev[ic++] = crs[3 + kshift]; slicev[ic++] = crs[4 + kshift];
            slicev[ic++] = crs[9 + kshift]; slicev[ic++] = crs[10 + kshift];
            slicev[ic++] = crs[6 + kshift]; slicev[ic++] = crs[7 + kshift];
            slicev[ic++] = crs[0 + kshift]; slicev[ic++] = crs[1 + kshift]; /*close*/

            ibv[ibn++] = ib;
        }

    }

    return ibn;
}
