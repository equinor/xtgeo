/*
 ***************************************************************************************
 *
 * NAME:
 *    grd3d_reverse_jrows.c
 *
 * DESCRIPTION:
 *    Reverting J axes, i.e. grid goes from right-handed to left-handed or vice-versa
 *
 * ARGUMENTS:
 *    nx, ny, nz       i     NCOL, NROW, NLAY dimens
 *    coordsv         i/o    Cornerlines, pillars w/ numpy dimensions
 *    zcornsv         i/o    ZCORN w/ numpy dimensions
 *    actnumsv        i/o    ACTNUM values w/ numpy dimensions
 *
 * RETURNS:
 *    Function: 0: upon success. Update pointers in-place
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ***************************************************************************************
 */
#include <stdlib.h>

#include <xtgeo/xtgeo.h>

#include "logger.h"

int
grd3d_reverse_jrows(int nx,
                    int ny,
                    int nz,
                    double *coordsv,
                    long ncoord,
                    double *zcornsv,
                    long nzcorn,
                    int *actnumsv,
                    long nact)

{

    long ic;
    int i, j, k;
    long ntot, ib1;
    double *tmpcv;
    int *itmp;

    logger_info(LI, FI, FU, "Flip/swap J axis");

    ntot = (nx + 1) * (ny + 1) * 6; /*  COORD values */
    tmpcv = calloc(ntot, sizeof(double));

    logger_info(LI, FI, FU, "J swapping COORD");
    logger_info(LI, FI, FU, "NX NY NZ %d %d %d", nx, ny, nz);
    for (j = 1; j <= ny + 1; j++) {
        for (i = 1; i <= nx + 1; i++) {

            int jx = ny + 1 - j + 1;

            tmpcv[6 * ((jx - 1) * (nx + 1) + i - 1) + 0] =
              coordsv[6 * ((j - 1) * (nx + 1) + i - 1) + 0];
            tmpcv[6 * ((jx - 1) * (nx + 1) + i - 1) + 1] =
              coordsv[6 * ((j - 1) * (nx + 1) + i - 1) + 1];
            tmpcv[6 * ((jx - 1) * (nx + 1) + i - 1) + 2] =
              coordsv[6 * ((j - 1) * (nx + 1) + i - 1) + 2];
            tmpcv[6 * ((jx - 1) * (nx + 1) + i - 1) + 3] =
              coordsv[6 * ((j - 1) * (nx + 1) + i - 1) + 3];
            tmpcv[6 * ((jx - 1) * (nx + 1) + i - 1) + 4] =
              coordsv[6 * ((j - 1) * (nx + 1) + i - 1) + 4];
            tmpcv[6 * ((jx - 1) * (nx + 1) + i - 1) + 5] =
              coordsv[6 * ((j - 1) * (nx + 1) + i - 1) + 5];
        }
    }
    for (ic = 0; ic < ntot; ic++)
        coordsv[ic] = tmpcv[ic];
    free(tmpcv);

    /*  ***************************************************************************** */
    logger_info(LI, FI, FU, "J swapping ZCORN");

    ntot = nx * ny * (nz + 1) * 4; /*  ZCORN values */
    tmpcv = calloc(ntot, sizeof(double));

    long ib2 = 0;
    for (k = 1; k <= nz + 1; k++) {
        for (j = ny; j >= 1; j--) {
            for (i = 1; i <= nx; i++) {
                ib1 = x_ijk2ib(i, j, k, nx, ny, nz + 1, 0);
                if (ib1 < 0) {
                    free(tmpcv);
                    throw_exception("Loop resulted in index outside "
                                    "boundary in grd3d_reverse_jrows");
                    return EXIT_FAILURE;
                }
                tmpcv[4 * ib2 + 0] = zcornsv[4 * ib1 + 2];
                tmpcv[4 * ib2 + 1] = zcornsv[4 * ib1 + 3];
                tmpcv[4 * ib2 + 2] = zcornsv[4 * ib1 + 0];
                tmpcv[4 * ib2 + 3] = zcornsv[4 * ib1 + 1];
                ib2++;
            }
        }
    }
    for (ic = 0; ic < ntot; ic++)
        zcornsv[ic] = tmpcv[ic];
    free(tmpcv);

    /*  ***************************************************************************** */
    logger_info(LI, FI, FU, "J swapping ACTNUM");

    ntot = nx * ny * nz; /*  ACTNUM values */
    itmp = calloc(ntot, sizeof(int));

    for (k = 1; k <= nz; k++) {
        for (j = 1; j <= ny; j++) {
            for (i = 1; i <= nx; i++) {
                ib1 = x_ijk2ib(i, j, k, nx, ny, nz, 0);
                ib2 = x_ijk2ib(i, ny - j + 1, k, nx, ny, nz, 0);
                if (ib1 < 0 || ib2 < 0) {
                    free(itmp);
                    throw_exception("Loop resulted in index outside "
                                    "boundary in grd3d_reverse_jrows");
                    return EXIT_FAILURE;
                }
                itmp[ib1] = actnumsv[ib2];
            }
        }
    }
    for (ic = 0; ic < ntot; ic++)
        actnumsv[ic] = itmp[ic];
    free(itmp);

    return EXIT_SUCCESS;
}
