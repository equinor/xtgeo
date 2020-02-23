/*
****************************************************************************************
 *
 * NAME:
 *    grd3d_calc_dxdy.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Computes the DX DY avg per cell
 *
 * ARGUMENTS:
 *    nx...nz        i     Dimensions
 *    p_coord_v      i     Coordinates (with size)
 *    p_zcorn_v      i     Z corners (with size)
 *    p_actnum_v     i     ACTNUM (with size)
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

#include "logger.h"
#include "libxtg.h"
#include "libxtg_.h"


int grd3d_calc_dxdy(
    int nx,
    int ny,
    int nz,
    double *p_coord_v,
    long ncoord,
    double *p_zcorn_v,
    long nzcorn,
    int *p_actnum_v,
    long nactnum,
    double *dx,
    long ndx,
    double *dy,
    long ndy,
    int option1,
    int option2
    )

{
    /* locals */
    int     i, j, k, n, ii;

    double  c[24], plen, vlen, arad, adeg;

    logger_info(LI, FI, FU, "Compute DX DY...");

    for (k = 1; k <= nz; k++) {
	for (j = 1; j <= ny; j++) {
	    for (i = 1; i <= nx; i++) {

		long ib = x_ijk2ib(i,j,k,nx,ny,nz,0);
		long ic = x_ijk2ic(i,j,k,nx,ny,nz,0);

                if (option1 == 1 && p_actnum_v[ib] == 0) {
                    dx[ic] = UNDEF;
                    dx[ic] = UNDEF;
                    continue;
                }

                grd3d_corners(i, j, k, nx, ny, nz,
                              p_coord_v, 0, p_zcorn_v, 0, c);

                /* get the length of all lines forming DX */
                plen = 0.0;
                for (n = 0; n <= 3; n++) {
                    ii = 0 + n*6;
                    x_vector_info2(c[ii], c[ii+3], c[ii+1], c[ii+4],
                                   &vlen, &arad, &adeg, 1, XTGDEBUG);
                    plen = plen + vlen;
                }
                dx[ic] = plen/4.0;

                /* get the length of all lines forming DY */
                plen = 0.0;
                for (n = 0; n <= 3; n++) {
                    ii = 0 + n*3;
                    if (n >= 2) ii = 6 + n*3;

                    x_vector_info2(c[ii], c[ii+6], c[ii+1], c[ii+7],
                                   &vlen, &arad, &adeg, 1, XTGDEBUG);
                    plen = plen + vlen;
                }
                dy[ic] = plen/4.0;
            }
        }
    }

    logger_info(LI, FI, FU, "Compute DX DY... done");
    return EXIT_SUCCESS;
}
