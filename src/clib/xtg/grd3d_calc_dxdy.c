/*
 ***************************************************************************************
 *
 * NAME:
 *    grd3d_calc_dxdy.c
 *
 * DESCRIPTION:
 *    Computes the DX DY avg per cell
 *
 * ARGUMENTS:
 *    nx...nz        i     Dimensions
 *     coordsv       i     Coordinates (with size)
 *    zcornsv        i     Z corners (with size)
 *    actnumsv       i     ACTNUM (with size)
 *    dx            i/o    Array to be updated
 *    dy            i/o    Array to be updated
 *    option1        i     If 1, set dx dy to UNDEF for inactive cells
 *    option2        i     Unused
 *
 * RETURNS:
 *    Success (0) or failure. Pointers to arrays are updated
 *
 * NOTES:
 *    The returned arrays are now C order
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ***************************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"
#include "logger.h"

int
grd3d_calc_dxdy(int nx,
                int ny,
                int nz,
                double *coordsv,
                long ncoord,
                double *zcornsv,
                long nzcorn,
                int *actnumsv,
                long nactnum,
                double *dx,
                long ndx,
                double *dy,
                long ndy,
                int option1,
                int option2)

{
    /* locals */
    int i, j, k, n, ii;

    double c[24], plen, vlen, arad, adeg;

    logger_info(LI, FI, FU, "Compute DX DY...");

    long ntot[3] = { nactnum, ndx, ndy };

    if (x_verify_vectorlengths(nx, ny, nz, ncoord, nzcorn, ntot, 3) != 0) {
        throw_exception("Errors in array lengths checks in grd3d_calc_dxdy");
        return EXIT_FAILURE;
    }
    if (option2 == 0)
        logger_debug(LI, FI, FU, "Option2 not in use");

    for (k = 1; k <= nz; k++) {
        for (j = 1; j <= ny; j++) {
            for (i = 1; i <= nx; i++) {

                long ib = x_ijk2ib(i, j, k, nx, ny, nz, 0);
                long ic = x_ijk2ic(i, j, k, nx, ny, nz, 0);

                if (option1 == 1 && actnumsv[ib] == 0) {
                    dx[ic] = UNDEF;
                    dy[ic] = UNDEF;
                    continue;
                }

                grd3d_corners(i, j, k, nx, ny, nz, coordsv, 0, zcornsv, 0, c);

                /* get the length of all lines forming DX */
                plen = 0.0;
                for (n = 0; n <= 3; n++) {
                    ii = 0 + n * 6;
                    x_vector_info2(c[ii], c[ii + 3], c[ii + 1], c[ii + 4], &vlen, &arad,
                                   &adeg, 1);
                    plen = plen + vlen;
                }
                dx[ic] = plen / 4.0;

                /* get the length of all lines forming DY */
                plen = 0.0;
                for (n = 0; n <= 3; n++) {
                    ii = 0 + n * 3;
                    if (n >= 2)
                        ii = 6 + n * 3;

                    x_vector_info2(c[ii], c[ii + 6], c[ii + 1], c[ii + 7], &vlen, &arad,
                                   &adeg, 1);
                    plen = plen + vlen;
                }
                dy[ic] = plen / 4.0;
            }
        }
    }

    logger_info(LI, FI, FU, "Compute DX DY... done");
    return EXIT_SUCCESS;
}
