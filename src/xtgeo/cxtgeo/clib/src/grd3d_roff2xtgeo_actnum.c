/*
 ***************************************************************************************
 *
 * Convert from ROFF grid cornerpoint spec to XTGeo cornerpoint grid: ACTNUM array
 *
 ***************************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"

/*
 ***************************************************************************************
 *
 * NAME:
 *    grd3d_roff2xtgeo_actnum.c
 *
 * DESCRIPTION:
 *    Convert fro ROFF internal spec to XTGeo spec for ACTNUM, The spec differs in
 *    ordering, where XTGeo is column major ala Eclipse.
 *
 * ARGUMENTS:
 *    nx, ny, nz       i     NCOL, NROW, NLAY dimens
 *    p_act_v          i     Input actnum array ROFF fmt
 *    p_actnum_v       o     Output actnum array XTGEO fmt
 *    debug            i     Debug level
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

int grd3d_roff2xtgeo_actnum (
                            int nx,
                            int ny,
                            int nz,
                            int *p_act_v,
                            int *p_actnum_v,
                            int debug
                            )

{

    long ib, ic, nact = 0;
    int i, j, k;

    char sbn[24] = "grd3d_roff2xtgeo_actnum";
    xtgverbose(debug);

    xtg_speak(sbn, 2, "Transforming grid ROFF actnum --> XTG representation ...");

    ic = 0;
    for (i = 0; i < nx; i++) {
        for (j = 0;j < ny; j++) {
            for (k = 0;k < nz; k++) {
                ib = (nz - (k + 1)) * ny * nx + j * nx + i;
                p_actnum_v[ib] = p_act_v[ic];
                ic += 1;
                if (p_actnum_v[ib] == 1) nact++;
            }
        }
    }

    return nact;
}
