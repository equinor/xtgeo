/*
 ***************************************************************************************
 *
 * Do reversing / flipping  of J coordinates in the grid
 *
 ***************************************************************************************
 */

#include "logger.h"
#include "libxtg.h"
#include "libxtg_.h"

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
 *    p_coord_v       i/o    Cornerlines, pillars
 *    p_zcorn_v       i/o    ZCORN
 *    p_actnum_v      i/o    ACTNUM values
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

int grd3d_reverse_jrows (
    int nx,
    int ny,
    int nz,
    double *p_coord_v,
    double *p_zcorn_v,
    int *p_actnum_v
    )

{

    long ic;
    int i, j, k;
    long ntot, ib1, ib2;
    double *tmpcv;
    int *itmp;

    logger_info(LI, FI, FU, "Flip/swap J axis");

    ntot = (nx + 1) * (ny + 1) * 6;  /*  COORD values */
    tmpcv = calloc(ntot, sizeof(double));

    logger_info(LI, FI, FU, "J swapping COORD");
    logger_info(LI, FI, FU, "NX NY NZ %d %d %d", nx, ny, nz);
    ib2 = 0;
    for (j = 1; j <= ny + 1; j++) {
        for (i = 1; i <= nx + 1; i++) {

            int jx = ny + 1 - j + 1;

            tmpcv[6*((jx-1)*(nx+1)+i-1)+0] = p_coord_v[6*((j-1)*(nx+1)+i-1)+0];
            tmpcv[6*((jx-1)*(nx+1)+i-1)+1] = p_coord_v[6*((j-1)*(nx+1)+i-1)+1];
            tmpcv[6*((jx-1)*(nx+1)+i-1)+2] = p_coord_v[6*((j-1)*(nx+1)+i-1)+2];
            tmpcv[6*((jx-1)*(nx+1)+i-1)+3] = p_coord_v[6*((j-1)*(nx+1)+i-1)+3];
            tmpcv[6*((jx-1)*(nx+1)+i-1)+4] = p_coord_v[6*((j-1)*(nx+1)+i-1)+4];
            tmpcv[6*((jx-1)*(nx+1)+i-1)+5] = p_coord_v[6*((j-1)*(nx+1)+i-1)+5];
        }
    }
    for (ic = 0; ic < ntot; ic++) p_coord_v[ic] = tmpcv[ic];
    free(tmpcv);

    /*  ***************************************************************************** */
    logger_info(LI, FI, FU, "J swapping ZCORN");

    ntot = nx * ny * (nz + 1) * 4;  /*  ZCORN values */
    tmpcv = calloc(ntot, sizeof(double));

    ib2 = 0;
    for (k = 1; k <= nz + 1; k++) {
        for (j = ny; j >= 1; j--) {
            for (i = 1; i <= nx; i++) {
                ib1 = x_ijk2ib(i, j, k, nx, ny, nz + 1, 0);
                tmpcv[4*ib2 + 0] = p_zcorn_v[4*ib1 + 2];
                tmpcv[4*ib2 + 1] = p_zcorn_v[4*ib1 + 3];
                tmpcv[4*ib2 + 2] = p_zcorn_v[4*ib1 + 0];
                tmpcv[4*ib2 + 3] = p_zcorn_v[4*ib1 + 1];
                ib2++;
            }
        }
    }
    for (ic = 0; ic < ntot; ic++) p_zcorn_v[ic] = tmpcv[ic];
    free(tmpcv);

    /*  ***************************************************************************** */
    logger_info(LI, FI, FU, "J swapping ACTNUM");

    ntot = nx * ny * nz;  /*  ACTNUM values */
    itmp = calloc(ntot, sizeof(int));


    for (k = 1; k <= nz; k++) {
        for (j = 1; j <= ny; j++) {
            for (i = 1; i <= nx; i++) {
                ib1 = x_ijk2ib(i, j, k, nx, ny, nz, 0);
                ib2 = x_ijk2ib(i, ny - j + 1, k, nx, ny, nz, 0);
                itmp[ib1] = p_actnum_v[ib2];
            }
        }
    }
    for (ic = 0; ic < ntot; ic++) p_actnum_v[ic] = itmp[ic];
    free(itmp);

    return EXIT_SUCCESS;
}
