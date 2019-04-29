/*
 *******************************************************************************
 *
 * Refine a grid vertically
 *
 *******************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"

/*
 *******************************************************************************
 *
 * NAME:
 *    grd3d_refine_vert.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Proportional refinement per layer
 *
 * ARGUMENTS:
 *    nx .. nz       i     Dimensions
 *    p_coord_v      i     Coordinates
 *    p_zcorn_v      i     Z corners input
 *    p_actnum_v     i     ACTNUM, input
 *    nzref          i     New NZ
 *    p_zcornref_v   o     Z corners output (must be allocated before)
 *    p_actnumref_v  o     ACTNUM, new (must be allocated)
 *    p_num_act      o     Updated number of active cells
 *    rfac           i     Array for refinement factor per layer
 *    option         i     Options flag for later usage
 *    debug          i     Debug level
 *
 * RETURNS:
 *    Function: 0: upon success
 *    Pointers to arrays are updated
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 *******************************************************************************
 */

int grd3d_refine_vert (
                       int nx,
                       int ny,
                       int nz,
                       double *p_coord_v,
                       double *p_zcorn_v,
                       int *p_actnum_v,
                       int nzref,
                       double *p_zcornref_v,
                       int *p_actnumref_v,
                       int *p_num_act,
                       int *rfac,
                       int option,
                       int debug
                       )

{
    /* locals */
    int i, j, k, ib, ic, kr, ibt, ibb, ibrt, ibrb, kk, rfactor, iact;
    double rdz, ztop, zbot;
    char s[24] = "grd3d_refine_vert";

    xtgverbose(debug);

    xtg_speak(s, 1, "Entering <%s>", s);

    grd3d_make_z_consistent(nx, ny, nz, p_zcorn_v, p_actnum_v, 0.0, debug);

    for (j = 1; j <= ny; j++) {
	for (i = 1; i <= nx; i++) {

            kk = 1; /* refined grid K counter */

	    for (k = 1; k <= nz; k++) {

		ibt = x_ijk2ib(i, j, k, nx, ny, nz + 1, 0);
		ibb = x_ijk2ib(i, j, k + 1, nx, ny, nz + 1, 0);

                /* look at each pillar in each cell, find top */
                /* and bottom, and divide */

                rfactor = rfac[k - 1];  /* as array is 0 base */

                ib = x_ijk2ib(i, j, k, nx, ny, nz, 0);
                iact = p_actnum_v[ib];

		for (ic=1;ic<=4;ic++) {
		    ztop = p_zcorn_v[4*ibt + 1*ic - 1];
		    zbot = p_zcorn_v[4*ibb + 1*ic - 1];

                    /* now divide and assign to new zcorn for refined: */
                    rdz = (zbot - ztop) / rfactor;

                    if (rdz < -1 * FLOATEPS) {
                        xtg_error(s, "STOP! negative cell thickness found at "
                                  "%d %d %d", i, j, k);
                        return(-9);
                    }
                    /* now assign corners for the refined grid: */
                    for (kr = 0; kr < rfactor; kr++) {
                        ibrt = x_ijk2ib(i, j, kk + kr, nx, ny, nzref + 1, 0);

                        ibrb = x_ijk2ib(i, j, kk + kr + 1, nx, ny,
                                        nzref + 1, 0);

                        ib = x_ijk2ib(i, j, kk + kr, nx, ny, nzref, 0);
                        p_actnumref_v[ib] = iact;

                        p_zcornref_v[4*ibrt + 1*ic - 1] = ztop + kr * rdz;
                        p_zcornref_v[4*ibrb + 1*ic - 1] = ztop + (kr+1) * rdz;

                    }
                }

                kk = kk + rfactor;
            }
        }
    }

    xtg_speak(s , 2, "Exit from <%s>", s);
    return(0);

}
