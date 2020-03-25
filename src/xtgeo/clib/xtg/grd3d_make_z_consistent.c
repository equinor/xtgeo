/*
 ***************************************************************************************
 *
 * NAME:
 *    grd3d_make_z_consistent.c
 *
 * DESCRIPTION:
 *    Some other functions will fail if z-nodes are inconsistent in depth (e.g.
 *    grd3_adj_z_from_map). This routine will make z consistent, and also add
 *    a (very) small separation (zsep) if the separation is too small
 *
 * ARGUMENTS:
 *    nx, ny, nz         i     Grid dimensions ncol, nrow, nlay
 *    zcornsv           i/o    Grid Z corners (with numpy dimensions)
 *    zsep               i     Minimum seperation distance
 *
 * RETURNS:
 *    Void function; updated zcornsv pointer
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ***************************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"
#include "logger.h"

void
grd3d_make_z_consistent(int nx,
                        int ny,
                        int nz,
                        double *zcornsv,
                        long nzcorn,
                        double zsep)

{

    logger_info(LI, FI, FU, "Entering %s with zsep %lf", FU, zsep);

    int i, j, k;

    for (j = 1; j <= ny; j++) {
        for (i = 1; i <= nx; i++) {
            for (k = 2; k <= nz + 1; k++) {

                long ibp = x_ijk2ib(i, j, k - 1, nx, ny, nz + 1, 0);
                long ibx = x_ijk2ib(i, j, k, nx, ny, nz + 1, 0);

                int ic;
                for (ic = 1; ic <= 4; ic++) {
                    double z1 = zcornsv[4 * ibp + 1 * ic - 1];
                    double z2 = zcornsv[4 * ibx + 1 * ic - 1];

                    if ((z2 - z1) < zsep) {
                        zcornsv[4 * ibx + 1 * ic - 1] = z1 + zsep;
                    }
                }
            }
        }
    }

    logger_info(LI, FI, FU, "Exit from %s", FU);
}
