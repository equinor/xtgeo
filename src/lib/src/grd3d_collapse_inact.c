/*
****************************************************************************************
*
* NAME:
*    grd3d_collapse_inact.c
*
* DESCRIPTION:
*    Collapse inactive cells
*
* ARGUMENTS:
*    nx, ny, nz     i     Dimensions
*    **sv           i     Geometry arrays (w numpy dimensions)
*
* RETURNS:
*    Void, update p_* array pointers
*
* TODO/ISSUES/BUGS:
*
* LICENCE:
*    CF XTGeo's LICENSE
***************************************************************************************
*/
#include <xtgeo/xtgeo.h>
#include "logger.h"

void
grd3d_collapse_inact(int nx,
                     int ny,
                     int nz,
                     double *zcornsv,
                     long nzcornin,
                     int *actnumsv,
                     long nactin)

{
    /* locals */
    int i, j, k, ic, ibp, ibx, iflag, kk, kkk, k2 = 0;
    double z1, z2;

    logger_info(LI, FI, FU, "Collapsing inactive cells...");

    for (j = 1; j <= ny; j++) {

        for (i = 1; i <= nx; i++) {
            iflag = 0;
            /* check that column has active cells */
            for (k = 2; k <= nz + 1; k++) {

                ibp = x_ijk2ib(i, j, k - 1, nx, ny, nz + 1, 0);
                if (actnumsv[ibp] == 1) {
                    iflag = 1;
                }
            }

            if (iflag == 1) {

                for (k = 2; k <= nz + 1; k++) {
                    ibp = x_ijk2ib(i, j, k - 1, nx, ny, nz + 1, 0);
                    /* find inactive cell */

                    if (actnumsv[ibp] == 0) {

                        /* find next active cell */
                        for (kk = k; kk <= nz + 1; kk++) {

                            if (kk < nz + 1) {
                                ibx = x_ijk2ib(i, j, kk, nx, ny, nz + 1, 0);

                                if (actnumsv[ibx] == 1) {
                                    k2 = kk;
                                    break;
                                }
                            }
                        }
                        /* check each corner */

                        ibx = x_ijk2ib(i, j, k2, nx, ny, nz + 1, 0);
                        for (ic = 1; ic <= 4; ic++) {
                            z1 = zcornsv[4 * ibp + 1 * ic - 1];
                            z2 = zcornsv[4 * ibx + 1 * ic - 1];
                            if ((z2 - z1) > 0.0) {
                                /* k-1 */
                                zcornsv[4 * ibp + 1 * ic - 1] = 0.5 * (z1 + z2);
                                /* all the other below */
                                for (kkk = k; kkk <= k2; kkk++) {
                                    ibx = x_ijk2ib(i, j, kkk, nx, ny, nz + 1, 0);
                                    zcornsv[4 * ibx + 1 * ic - 1] = 0.5 * (z1 + z2);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    logger_info(LI, FI, FU, "Collapsing inactive cells... done");
}
