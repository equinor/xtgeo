/*
 ***************************************************************************************
 *
 * NAME:
 *    grd3d_calc_dz.c
 *
 *
 * DESCRIPTION:
 *    Computes the DZ avg per cell
 *
 * ARGUMENTS:
 *    nx...nz        i     Dimensions
 *    zcornsv        i     Z corners (with size)
 *    actnumsv       i     ACTNUM (with size)
 *    p_dz_v        i/o    Array to be updated
 *    flip           i     Vertical flip flag
 *    option         i     0: all cells, 1: make ACTNUM  UNDEF
 *
 * RETURNS:
 *    Void. Pointers to arrays are updated
 *
 * NOTES:
 *    The returned array is now C order
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ***************************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"
#include "logger.h"

void
grd3d_calc_dz(int nx,
              int ny,
              int nz,
              double *zcornsv,
              long nzcorn,
              int *actnumsv,
              long nactnum,
              double *p_dz_v,
              long ndz,
              int flip,
              int option)

{
    int i, j, k;
    double top_z_avg, bot_z_avg;

    logger_info(LI, FI, FU, "Compute DZ...");

    for (k = 1; k <= nz; k++) {
        for (j = 1; j <= ny; j++) {
            for (i = 1; i <= nx; i++) {

                /* parameter counting */
                long ic = x_ijk2ic(i, j, k, nx, ny, nz, 0); /* C order */
                long ib = x_ijk2ib(i, j, k, nx, ny, nz, 0); /* F order */
                if (ic < 0 || ib < 0) {
                    throw_exception("Loop through grid resulted in index outside "
                                    "boundary in grd3d_calc_dz");
                    return;
                }
                /* grid */
                long ip = x_ijk2ib(i, j, k, nx, ny, nz + 1, 0);
                long iq = x_ijk2ib(i, j, k + 1, nx, ny, nz + 1, 0);
                if (ic < 0 || ib < 0) {
                    throw_exception("Index outside boundary in grd3d_calc_dz, trying "
                                    "to access non-existing layer");
                    return;
                }
                /* each cell */
                top_z_avg = 0.25 * (zcornsv[4 * ip + 1 - 1] + zcornsv[4 * ip + 2 - 1] +
                                    zcornsv[4 * ip + 3 - 1] + zcornsv[4 * ip + 4 - 1]);
                bot_z_avg = 0.25 * (zcornsv[4 * iq + 1 - 1] + zcornsv[4 * iq + 2 - 1] +
                                    zcornsv[4 * iq + 3 - 1] + zcornsv[4 * iq + 4 - 1]);

                p_dz_v[ic] = (double)flip * (bot_z_avg - top_z_avg);
                // will do it correct for flipped grids

                if (option == 1 && actnumsv[ib] == 0) {
                    p_dz_v[ic] = UNDEF;
                }
            }
        }
    }
    logger_info(LI, FI, FU, "Compute DZ... done");
}
