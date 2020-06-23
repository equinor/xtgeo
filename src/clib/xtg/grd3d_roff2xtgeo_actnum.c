/*
 ***************************************************************************************
 *
 * NAME:
 *    grd3d_roff2xtgeo_actnum.c
 *
 * DESCRIPTION:
 *    Convert from ROFF internal spec to XTGeo spec for ACTNUM, The spec differs in
 *    ordering, where XTGeo is column major ala Eclipse.
 *
 * ARGUMENTS:
 *    nx, ny, nz       i     NCOL, NROW, NLAY dimens
 *    p_act_v          i     Input actnum array ROFF fmt
 *    actnumsv         o     Output actnum array XTGEO fmt
 *    option           i     If 1, then all cells shall be regarded as active
 *
 * RETURNS:
 *    Number of active cells
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

int
grd3d_roff2xtgeo_actnum(int nx,
                        int ny,
                        int nz,
                        int *p_act_v,
                        int *actnumsv,
                        long nactnum,
                        int option)

{

    long ib = 0, ic = 0, nact = 0;
    int i, j, k;

    logger_info(LI, FI, FU, "Transforming grid ROFF actnum -> XTG representation ...");

    if (option == 1) {
        for (ib = 0; ib < nx * ny * nz; ib++) {
            actnumsv[ib] = 1;
        }
        return nx * ny * nz;
    }

    ic = 0;
    for (i = 0; i < nx; i++) {
        for (j = 0; j < ny; j++) {
            for (k = 0; k < nz; k++) {
                ib = (nz - (k + 1)) * ny * nx + j * nx + i;
                actnumsv[ib] = p_act_v[ic];
                ic += 1;
                if (actnumsv[ib] == 1)
                    nact++;
            }
        }
    }

    logger_info(LI, FI, FU, "Transforming grid ROFF actnum -> XTG representation done");

    return nact;
}
