/*
 ***************************************************************************************
 *
 * NAME:
 *    grdcp3d_calc_xyz.c
 *
 * DESCRIPTION:
 *    Get X Y Z vector per cell
 *
 * ARGUMENTS:
 *     nx..nz           grid dimensions
 *     coordsv          Coordinates
 *     zcornsv          ZCORN array (pointer) of input
 *     p_x_v .. p_z_v   Return arrays for X Y Z
 *     option           0: use all cells, 1: make undef if ACTNUM=0
 *
 * RETURNS:
 *    Void, update arrays
 *
 * TODO/ISSUES/BUGS:
 *    Make proper return codes
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ***************************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"

void
grdcp3d_calc_xyz(long ncol,
               long nrow,
               long nlay,
               double *coordsv,
               long ncoord,
               float *zcornsv,
               long nzcorn,
               int *actnumsv,
               long nact,
               int option,
               double *xarr,  // xarr
               long nxarr,
               double *yarr,  // yarr
               long nyarr,
               double *zarr,  // zarr
               long nzarr)
{
    long ntotv[4] = { nact, nxarr, nyarr, nzarr };
    if (x_verify_vectorlengths(ncol, nrow, nlay, ncoord, nzcorn, ntotv, 4,
                XTGFORMAT2) != 0) {
        throw_exception("Errors in array lengths checks in grdcp3d_calc_xyz");
        return;
    }

    double xv, yv, zv;
    for (long i = 0; i < ncol; i++) {
        for (long j = 0; j < nrow; j++) {
            for (long k = 0; k < nlay; k++) {
                /* Offset (i,j,k) by 1 because starting from 0. */
                long ic = x_ijk2ic(i + 1, j + 1, k + 1, ncol, nrow, nlay, 0);
                if (ic < 0) {
                    throw_exception("Loop through grid resulted in index outside "
                                    "boundary in grdcp3d_calc_xyz");
                    return;
                }

                /* If we want to mask inactive cells */
                if (option == 1 && actnumsv[ic] == 0) {
                    xarr[ic] = UNDEF;
                    yarr[ic] = UNDEF;
                    zarr[ic] = UNDEF;
                    continue;
                }

                grdcp3d_midpoint(i, j, k, ncol, nrow, nlay, coordsv, ncoord, zcornsv,
                        nzcorn, &xv, &yv, &zv);

                xarr[ic] = xv;
                yarr[ic] = yv;
                zarr[ic] = zv;
            }
        }
    }
}
