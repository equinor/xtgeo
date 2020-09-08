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
***************************************************************************************
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
    long icn, jcn;
    for (icn = 0; icn < nncol; icn++) {
        for (jcn = 0; jcn < nnrow; jcn++) {
            int nn;
            for (nn = 0; nn < 3; nn++) {
                tpillars[ic + nn] = coordsv[ic + nn];
                bpillars[ic + nn] = coordsv[ic + nn + 3];
            }
            ic = ic + 3;
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
    long icol, jrow, klay;
    for (icol = 0; icol < nncol; icol++) {
        for (jrow = 0; jrow < nnrow; jrow++) {
            for (klay = 0; klay < nnlay; klay++) {

                double z0 = 4 * (icol * nnrow * nnlay + jrow * nnlay + klay) + 0;
                double z1 = 4 * (icol * nnrow * nnlay + jrow * nnlay + klay) + 1;
                double z2 = 4 * (icol * nnrow * nnlay + jrow * nnlay + klay) + 2;
                double z3 = 4 * (icol * nnrow * nnlay + jrow * nnlay + klay) + 3;

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
