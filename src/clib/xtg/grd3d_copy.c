/*
 ***************************************************************************************
 *
 * NAME:
 *    grd3d_copy.c
 *
 * DESCRIPTION:
 *    Copy from input pointer arrays to new
 *
 * ARGUMENTS:
 *    ncol,nrow,nlay  i     Grid dimensions IX JY KZ in input
 *    p_coord1_v      i     Grid Z coord for input
 *    p_zcorn1_v      i     Grid Z corners for input
 *    p_actnum1_v     i     Grid ACTNUM parameter input
 *    p_coord2_v      o     Grid Z coord for output
 *    p_zcorn2_v      o     Grid Z corners for output
 *    p_actnum2_v     o     Grid ACTNUM parameter output
 *    iflag           i     Options flag (future use)
 *
 * RETURNS:
 *    The C macro EXIT_SUCCESS unless problems + changed pointers
 *
 * TODO/ISSUES/BUGS:
 *    None known
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ***************************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"
#include "logger.h"

int
grd3d_copy(int ncol,
           int nrow,
           int nlay,
           double *p_coord1_v,
           double *p_zcorn1_v,
           int *p_actnum1_v,
           double *p_coord2_v,
           double *p_zcorn2_v,
           int *p_actnum2_v,
           int iflag)
{
    /* locals */
    int ic, icn, jcn, kcn;
    long ijo, ibt;

    for (kcn = 1; kcn <= nlay + 1; kcn++) {
        for (jcn = 1; jcn <= nrow; jcn++) {
            for (icn = 1; icn <= ncol; icn++) {

                if (kcn == 1) {

                    ijo = 6 * ((jcn - 1) * (ncol + 1) + icn - 1);

                    p_coord2_v[ijo + 0] = p_coord1_v[ijo + 0];
                    p_coord2_v[ijo + 1] = p_coord1_v[ijo + 1];
                    p_coord2_v[ijo + 2] = p_coord1_v[ijo + 2];
                    p_coord2_v[ijo + 3] = p_coord1_v[ijo + 3];
                    p_coord2_v[ijo + 4] = p_coord1_v[ijo + 4];
                    p_coord2_v[ijo + 5] = p_coord1_v[ijo + 5];

                    if (jcn <= nrow && icn == ncol) {

                        ijo = 6 * ((jcn - 1) * (ncol + 1) + icn - 0);

                        p_coord2_v[ijo + 0] = p_coord1_v[ijo + 0];
                        p_coord2_v[ijo + 1] = p_coord1_v[ijo + 1];
                        p_coord2_v[ijo + 2] = p_coord1_v[ijo + 2];
                        p_coord2_v[ijo + 3] = p_coord1_v[ijo + 3];
                        p_coord2_v[ijo + 4] = p_coord1_v[ijo + 4];
                        p_coord2_v[ijo + 5] = p_coord1_v[ijo + 5];
                    }

                    if (jcn == nrow && icn <= ncol) {

                        ijo = 6 * ((jcn - 0) * (ncol + 1) + icn - 1);

                        p_coord2_v[ijo + 0] = p_coord1_v[ijo + 0];
                        p_coord2_v[ijo + 1] = p_coord1_v[ijo + 1];
                        p_coord2_v[ijo + 2] = p_coord1_v[ijo + 2];
                        p_coord2_v[ijo + 3] = p_coord1_v[ijo + 3];
                        p_coord2_v[ijo + 4] = p_coord1_v[ijo + 4];
                        p_coord2_v[ijo + 5] = p_coord1_v[ijo + 5];
                    }

                    if (jcn == nrow && icn == ncol) {

                        ijo = 6 * ((jcn - 0) * (ncol + 1) + icn - 0);

                        p_coord2_v[ijo + 0] = p_coord1_v[ijo + 0];
                        p_coord2_v[ijo + 1] = p_coord1_v[ijo + 1];
                        p_coord2_v[ijo + 2] = p_coord1_v[ijo + 2];
                        p_coord2_v[ijo + 3] = p_coord1_v[ijo + 3];
                        p_coord2_v[ijo + 4] = p_coord1_v[ijo + 4];
                        p_coord2_v[ijo + 5] = p_coord1_v[ijo + 5];
                    }
                }

                ibt = x_ijk2ib(icn, jcn, kcn, ncol, nrow, nlay + 1, 0);
                if (ibt < 0) {
                    throw_exception("Loop through grid resulted in index outside "
                                    "boundary in grd3d_copy");
                    return EXIT_FAILURE;
                }

                for (ic = 1; ic <= 4; ic++) {
                    p_zcorn2_v[4 * ibt + 1 * ic - 1] = p_zcorn1_v[4 * ibt + 1 * ic - 1];
                }

                if (kcn <= nlay) {
                    ibt = x_ijk2ib(icn, jcn, kcn, ncol, nrow, nlay, 0);
                    if (ibt < 0) {
                        throw_exception("Loop through grid resulted in index outside "
                                        "boundary in grd3d_copy");
                        return EXIT_FAILURE;
                    }

                    p_actnum2_v[ibt] = p_actnum1_v[ibt];
                }
            }
        }
    }

    return EXIT_SUCCESS;
}
