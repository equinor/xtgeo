/*
****************************************************************************************
*
* NAME:
*    grdcp3d_conv_grid_roxapi.c
*
* DESCRIPTION:
*   Transform to ROXAR API internal format from xtgformat==2. Thes formats have many
*   similarities, so transform shall be easy and fast
*
* ARGUMENTS:
*    ncol, ..nlay   i     NCOL, NROW, NLAY
*    coordsv        i     COORD array
*    zcornsv        i     ZCORN array
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
****************************************************************************************
*/

#include "libxtg.h"
#include "libxtg_.h"
#include "logger.h"

int
grdcp3d_conv_grid_roxapi(long ncol,
                         long nrow,
                         long nlay,

                         double *coordsv,
                         long ncoordin,
                         float *zcornsv,
                         long nzcornin,

                         double *tpillars,
                         long ntpillars,
                         double *bpillars,
                         long nbpillars,
                         double *zcorners,
                         long nzcorners)

{

    logger_info(LI, FI, FU, "From XTGeo grid xtgversion==2 to ROXAPI grid...");

    /*
     *----------------------------------------------------------------------------------
     * COORD --> pillars
     *----------------------------------------------------------------------------------
     */
    long nncol = ncol + 1;
    long nnrow = nrow + 1;
    long nnlay = nlay + 1;

    long ic = 0;
    long ict = 0;
    long icb = 0;
    long icn, jcn;
    for (icn = 0; icn < nncol; icn++) {
        for (jcn = 0; jcn < nnrow; jcn++) {
            int nn;
            for (nn = 0; nn < 6; nn++) {
                if (nn < 3) {
                    tpillars[ict++] = coordsv[ic++];
                } else {
                    bpillars[icb++] = coordsv[ic++];
                }
            }
        }
    }

    /*
     *----------------------------------------------------------------------------------
     * ZCORN -- zcorners per pillar
     *----------------------------------------------------------------------------------
     *    nw  |  ne
     *      z2|z3       This is the ROXAPI per pillar, where 4 cells
     *   ------------   meet... E.g. means that z0 and z1 is UNDEF for i==0 etc
     *      z0|z1       (too be masked in ROXAPI python)
     *    sw  |  se
     *
     */

    ic = 0;
    long iz = 0;
    long icol, jrow, klay;

    for (icol = 0; icol < nncol; icol++) {
        for (jrow = 0; jrow < nnrow; jrow++) {

            double pz0 = VERYLARGENEGATIVE;
            double pz1 = VERYLARGENEGATIVE;
            double pz2 = VERYLARGENEGATIVE;
            double pz3 = VERYLARGENEGATIVE;

            for (klay = 0; klay < nnlay; klay++) {

                double z0 = (double)zcornsv[iz++];
                double z1 = (double)zcornsv[iz++];
                double z2 = (double)zcornsv[iz++];
                double z3 = (double)zcornsv[iz++];

                /* avoid depths that crosses in depth */
                if (z0 < pz0)
                    z0 = pz0;
                if (z1 < pz1)
                    z1 = pz1;
                if (z2 < pz2)
                    z2 = pz2;
                if (z3 < pz3)
                    z3 = pz3;

                pz0 = z0;
                pz1 = z1;
                pz2 = z2;
                pz3 = z3;

                if (icol == 0) {
                    z0 = UNDEF;
                    z2 = UNDEF;
                }
                if (icol == ncol) {
                    z1 = UNDEF;
                    z3 = UNDEF;
                }
                if (jrow == 0) {
                    z0 = UNDEF;
                    z1 = UNDEF;
                }
                if (jrow == nrow) {
                    z2 = UNDEF;
                    z3 = UNDEF;
                }

                zcorners[ic++] = z0;
                zcorners[ic++] = z1;
                zcorners[ic++] = z2;
                zcorners[ic++] = z3;
            }
        }
    }

    logger_info(LI, FI, FU, "From XTGeo grid xtgversion==2 to ROXAPI grid... done");

    return EXIT_SUCCESS;
}
