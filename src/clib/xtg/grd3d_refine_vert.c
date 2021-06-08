
/*
 ***************************************************************************************
 *
 * NAME:
 *    grd3d_refine_vert.c
 *
 *
 * DESCRIPTION:
 *    Proportional refinement per layer
 *
 * ARGUMENTS:
 *    nx .. nz       i     Dimensions
 *    coordsv        i     Coordinates
 *    zcornsv        i     Z corners input
 *    actnumsv       i     ACTNUM, input
 *    nzref          i     New NZ
 *    p_zcornref_v   o     Z corners output (must be allocated before)
 *    p_actnumref_v  o     ACTNUM, new (must be allocated)
 *    rfac           i     Array for refinement factor per layer
 *
 * RETURNS:
 *    Function: 0: upon success
 *    Pointers to arrays are updated
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
grd3d_refine_vert(int nx,
                  int ny,
                  int nz,

                  double *zcornsv,
                  long nzcorn,
                  int *actnumsv,
                  long nact,

                  int nzref,

                  double *p_zcornref_v,
                  long nzcornref,
                  int *p_actnumref_v,
                  long numactref,

                  int *rfac)

{
    /* locals */
    int i, j, k, ic, kr, kk, rfactor, iact;
    double rdz, ztop, zbot;

    grd3d_make_z_consistent(nx, ny, nz, zcornsv, 0, 0.0);

    for (j = 1; j <= ny; j++) {
        for (i = 1; i <= nx; i++) {

            kk = 1; /* refined grid K counter */

            for (k = 1; k <= nz; k++) {

                long ibt = x_ijk2ib(i, j, k, nx, ny, nz + 1, 0);
                long ibb = x_ijk2ib(i, j, k + 1, nx, ny, nz + 1, 0);
                if (ibt < 0 || ibb < 0) {
                    throw_exception("Index outside boundary in grd3d_refine_vert");
                    return EXIT_FAILURE;
                }

                /* look at each pillar in each cell, find top */
                /* and bottom, and divide */

                rfactor = rfac[k - 1]; /* as array is 0 base */

                long ib = x_ijk2ib(i, j, k, nx, ny, nz, 0);
                if (ib < 0) {
                    throw_exception("Index outside boundary in grd3d_refine_vert");
                    return EXIT_FAILURE;
                }
                iact = actnumsv[ib];

                for (ic = 1; ic <= 4; ic++) {
                    ztop = zcornsv[4 * ibt + 1 * ic - 1];
                    zbot = zcornsv[4 * ibb + 1 * ic - 1];

                    /* now divide and assign to new zcorn for refined: */
                    rdz = (zbot - ztop) / rfactor;

                    if (rdz < -1 * FLOATEPS) {
                        logger_error(LI, FI, FU,
                                     "STOP! negative cell thickness found "
                                     "at %d %d %d",
                                     i, j, k);
                        return (-9);
                    }
                    /* now assign corners for the refined grid: */
                    for (kr = 0; kr < rfactor; kr++) {
                        long ibrt = x_ijk2ib(i, j, kk + kr, nx, ny, nzref + 1, 0);
                        long ibrb = x_ijk2ib(i, j, kk + kr + 1, nx, ny, nzref + 1, 0);
                        if (ibrt < 0 || ibrb < 0) {
                            throw_exception(
                              "Index outside boundary in grd3d_refine_vert");
                            return EXIT_FAILURE;
                        }

                        ib = x_ijk2ib(i, j, kk + kr, nx, ny, nzref, 0);
                        if (ib < 0) {
                            throw_exception(
                              "Index outside boundary in grd3d_refine_vert");
                            return EXIT_FAILURE;
                        }

                        p_actnumref_v[ib] = iact;

                        p_zcornref_v[4 * ibrt + 1 * ic - 1] = ztop + kr * rdz;
                        p_zcornref_v[4 * ibrb + 1 * ic - 1] = ztop + (kr + 1) * rdz;
                    }
                }

                kk = kk + rfactor;
            }
        }
    }

    return EXIT_SUCCESS;
}
