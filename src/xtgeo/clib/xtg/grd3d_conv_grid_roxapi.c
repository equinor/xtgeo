/*
****************************************************************************************
*
* NAME:
*    grd3d_conv_grid_roxapi.c
*
* DESCRIPTION:
*   Transform to ROXAR API internal format.
*
* ARGUMENTS:
*    ncol, ..nlay   i     NCOL, NROW, NLAY
*    p_coord_v      i     COORD array
*    p_zcorn_v      i     ZCORN array
*    p_actnum_v     i     ACTNUM array
*    tpillars       o     Top node pillars (ncol+1 * nrow+1)*3, nrow fastest
*    bpillars       o     Base node pillars (ncol+1 * nrow+1)*3, nrow fastest
*    zcorners       o     Z value pillars, 4 per pillar, per depth
*    debug          i     Debug level
*
* RETURNS:
*    0: upon success
*
* LICENCE:
*    CF. XTGeo license
***************************************************************************************
*/

#include "libxtg.h"
#include "libxtg_.h"
#include "logger.h"

int
grd3d_conv_grid_roxapi(int ncol,
                       int nrow,
                       int nlay,

                       double *p_coord_v,
                       long ncoordin,
                       double *p_zcorn_v,
                       long nzcornin,
                       int *p_actnum_v,
                       long nactin,

                       double *tpillars,
                       long ntpillars,
                       double *bpillars,
                       long nbpillars,
                       double *zcorners,
                       long nzcorners)

{

    long ntotv[1] = { nactin };
    if (x_verify_vectorlengths(ncol, nrow, nlay, ncoordin, nzcornin, ntotv, 1) != 0)
        logger_critical(LI, FI, FU, "Bug: Errors in array lengths checks in %s", FU);

    long ic, ib, ib0, ib1, ib2, ib3;
    int icn, jcn, nn, i = 0, j = 0, k = 0;
    double z0, z1, z2, z3;

    logger_info(LI, FI, FU, "From XTGeo grid to ROXAPI grid...");

    /*
     *----------------------------------------------------------------------------------
     * COORD --> pillars
     *----------------------------------------------------------------------------------
     */

    ic = 0;
    for (icn = 1; icn <= ncol + 1; icn++) {
        for (jcn = 1; jcn <= nrow + 1; jcn++) { /* J (rows) cycling fastest */
            ib = 6 * ((jcn - 1) * (ncol + 1) + icn - 1);
            for (nn = 0; nn < 3; nn++) {
                tpillars[ic + nn] = p_coord_v[ib + nn];
                bpillars[ic + nn] = p_coord_v[ib + nn + 3];
            }
            ic = ic + 3;
        }
    }

    /*
     *----------------------------------------------------------------------------------
     * ZCORN -- zcorners per pillar
     *----------------------------------------------------------------------------------
     * XTGeo format having 4 corners pr cell layer as
     *                        3_____4
     *                        |     |
     *                        |     |
     *                        |_____|
     *                        1     2
     *        |
     *      z2|z3       While this is the ROXAPIR per pillar, where 4 cells
     *   ------------   meet... E.g. means that z0 and z1 is UNDEF for i==1
     *      z0|z1       (too be masked in ROXAPI python)
     *        |
     *
     */

    ic = 0;
    for (i = 1; i <= ncol + 1; i++) {
        for (j = 1; j <= nrow + 1; j++) {
            for (k = 1; k <= nlay + 1; k++) {

                ib0 = x_ijk2ib(i - 1, j - 1, k, ncol, nrow, nlay + 1, 0);
                ib1 = x_ijk2ib(i, j - 1, k, ncol, nrow, nlay + 1, 0);
                ib2 = x_ijk2ib(i - 1, j, k, ncol, nrow, nlay + 1, 0);
                ib3 = x_ijk2ib(i, j, k, ncol, nrow, nlay + 1, 0);

                z0 = UNDEF;
                z1 = UNDEF;
                z2 = UNDEF;
                z3 = UNDEF;

                if (i == 1 && j == 1) {
                    z3 = p_zcorn_v[4 * ib3 + 1 * 1 - 1];
                } else if (i == 1 && j == nrow + 1) {
                    z1 = p_zcorn_v[4 * ib1 + 1 * 3 - 1];
                } else if (i == ncol + 1 && j == 1) {
                    z2 = p_zcorn_v[4 * ib2 + 1 * 2 - 1];
                } else if (i == ncol + 1 && j == nrow + 1) {
                    z0 = p_zcorn_v[4 * ib0 + 1 * 4 - 1];
                } else if (i == 1) {
                    z1 = p_zcorn_v[4 * ib1 + 1 * 3 - 1];
                    z3 = p_zcorn_v[4 * ib3 + 1 * 1 - 1];
                } else if (i == ncol + 1) {
                    z0 = p_zcorn_v[4 * ib0 + 1 * 4 - 1];
                    z2 = p_zcorn_v[4 * ib2 + 1 * 2 - 1];
                } else if (j == 1) {
                    z2 = p_zcorn_v[4 * ib2 + 1 * 2 - 1];
                    z3 = p_zcorn_v[4 * ib3 + 1 * 1 - 1];
                } else if (j == nrow + 1) {
                    z1 = p_zcorn_v[4 * ib1 + 1 * 3 - 1];
                    z0 = p_zcorn_v[4 * ib0 + 1 * 4 - 1];
                } else {
                    z0 = p_zcorn_v[4 * ib0 + 1 * 4 - 1];
                    z1 = p_zcorn_v[4 * ib1 + 1 * 3 - 1];
                    z2 = p_zcorn_v[4 * ib2 + 1 * 2 - 1];
                    z3 = p_zcorn_v[4 * ib3 + 1 * 1 - 1];
                }

                zcorners[ic++] = z0;
                zcorners[ic++] = z1;
                zcorners[ic++] = z2;
                zcorners[ic++] = z3;
            }
        }
    }

    logger_info(LI, FI, FU, "From XTGeo grid to ROXAPI grid... done");

    return EXIT_SUCCESS;
}
