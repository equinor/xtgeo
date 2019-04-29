/*
 ******************************************************************************
 *
 * NAME:
 *    grd3d_crop_geometry.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Reduce the grid by cropping in I, J, K
 *
 * ARGUMENTS:
 *    nx,ny,nz       i     Grid dimensions I J K in input
 *    p_coord1_v     i     Grid Z coord for input
 *    p_zcorn1_v     i     Grid Z corners for input
 *    p_actnum1_v    i     Grid ACTNUM parameter input
 *    p_coord2_v     o     Grid Z coord for output
 *    p_zcorn2_v     o     Grid Z corners for output
 *    p_actnum2_v    o     Grid ACTNUM parameter output
 *    ic1, ... kc2   i     Crop from-to (inclusive)
 *    nactive        o     Number of active cells
 *    iflag          i     Options flag (future use)
 *    debug          i     Debug level
 *
 * RETURNS:
 *    The C macro EXIT_SUCCESS unless problems + changed pointers
 *
 * TODO/ISSUES/BUGS:
 *    None known
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ******************************************************************************
 */


#include "libxtg.h"
#include "libxtg_.h"


int grd3d_crop_geometry (
                         int nx,
                         int ny,
                         int nz,
                         double *p_coord1_v,
                         double *p_zcorn1_v,
                         int *p_actnum1_v,
                         double *p_coord2_v,
                         double *p_zcorn2_v,
                         int *p_actnum2_v,
                         int ic1, int ic2, int jc1, int jc2, int kc1, int kc2,
                         int *nactive,
                         int iflag,
                         int debug
                         )
{
    /* locals */
    char sbn[24] = "grd3d_crop_geometry";
    int newnx, newny, newnz, ic, icn, jcn, kcn, ncc, ixn, jxn, kxn;
    long ibb, ibt, ijo;

    xtgverbose(debug);
    xtg_speak(sbn, 1, "Entering routine <%s>", sbn);

    newnx = ic2 - ic1 + 1;
    newny = jc2 - jc1 + 1;
    newnz = kc2 - kc1 + 1;

    ibt=0;
    ncc=0;

    xtg_speak(sbn, 2, "Remapping COORDS, ZCORNs ACNUMs...");
    for (kcn = kc1; kcn <= kc2 + 1; kcn++) {
        for (jcn = jc1; jcn <= jc2; jcn++) {
            for (icn = ic1; icn <= ic2; icn++) {

                ixn = icn - ic1 + 1;
                jxn = jcn - jc1 + 1;
                kxn = kcn - kc1 + 1;

                if (kcn == kc1) {

                    ijo = 6 * ((jcn - 1) * (nx + 1) + icn - 1);
                    ibt = 6 * ((jxn - 1) * (newnx + 1) + ixn - 1);

                    p_coord2_v[ibt + 0] = p_coord1_v[ijo + 0];
                    p_coord2_v[ibt + 1] = p_coord1_v[ijo + 1];
                    p_coord2_v[ibt + 2] = p_coord1_v[ijo + 2];
                    p_coord2_v[ibt + 3] = p_coord1_v[ijo + 3];
                    p_coord2_v[ibt + 4] = p_coord1_v[ijo + 4];
                    p_coord2_v[ibt + 5] = p_coord1_v[ijo + 5];

                    if (jcn <= jc2 && icn == ic2) {

                        ijo = 6 * ((jcn - 1) * (nx + 1) + icn - 0);
                        ibt = 6 * ((jxn - 1) * (newnx + 1) + ixn - 0);

                        p_coord2_v[ibt + 0] = p_coord1_v[ijo + 0];
                        p_coord2_v[ibt + 1] = p_coord1_v[ijo + 1];
                        p_coord2_v[ibt + 2] = p_coord1_v[ijo + 2];
                        p_coord2_v[ibt + 3] = p_coord1_v[ijo + 3];
                        p_coord2_v[ibt + 4] = p_coord1_v[ijo + 4];
                        p_coord2_v[ibt + 5] = p_coord1_v[ijo + 5];
                    }

                    if (jcn == jc2 && icn <= ic2) {

                        ijo = 6 * ((jcn - 0) * (nx + 1) + icn - 1);
                        ibt = 6 * ((jxn - 0) * (newnx + 1) + ixn - 1);

                        p_coord2_v[ibt + 0] = p_coord1_v[ijo + 0];
                        p_coord2_v[ibt + 1] = p_coord1_v[ijo + 1];
                        p_coord2_v[ibt + 2] = p_coord1_v[ijo + 2];
                        p_coord2_v[ibt + 3] = p_coord1_v[ijo + 3];
                        p_coord2_v[ibt + 4] = p_coord1_v[ijo + 4];
                        p_coord2_v[ibt + 5] = p_coord1_v[ijo + 5];
                    }

                    if (jcn == jc2 && icn == ic2) {

                        ijo = 6 * ((jcn - 0) * (nx + 1) + icn - 0);
                        ibt = 6 * ((jxn - 0) * (newnx + 1) + ixn - 0);

                        p_coord2_v[ibt + 0] = p_coord1_v[ijo + 0];
                        p_coord2_v[ibt + 1] = p_coord1_v[ijo + 1];
                        p_coord2_v[ibt + 2] = p_coord1_v[ijo + 2];
                        p_coord2_v[ibt + 3] = p_coord1_v[ijo + 3];
                        p_coord2_v[ibt + 4] = p_coord1_v[ijo + 4];
                        p_coord2_v[ibt + 5] = p_coord1_v[ijo + 5];

                    }
                }


                ibt = x_ijk2ib(icn, jcn, kcn, nx, ny, nz + 1, 0);

                ibb = x_ijk2ib(icn - ic1 + 1, jcn - jc1 + 1, kcn - kc1 + 1,
                               newnx, newny, newnz + 1, 0);

                for (ic=1;ic<=4;ic++) {
                    p_zcorn2_v[4 * ibb + 1 * ic - 1] =
                        p_zcorn1_v[4 * ibt + 1 * ic - 1];
                }

                if (kcn <= kc2) {
                    ibt = x_ijk2ib(icn, jcn, kcn, nx, ny, nz, 0);

                    ibb = x_ijk2ib(icn - ic1 + 1, jcn - jc1 + 1, kcn - kc1 + 1,
                                   newnx, newny, newnz, 0);

                    p_actnum2_v[ibb] = p_actnum1_v[ibt];

                    if (p_actnum2_v[ibb] == 1) ncc++;
                }
            }
	}
    }

    *nactive=ncc;

    xtg_speak(sbn, 1, "Exit from <%s>", sbn);

    return EXIT_SUCCESS;

}
