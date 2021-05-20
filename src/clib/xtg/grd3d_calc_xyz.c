/*
 ***************************************************************************************
 *
 * NAME:
 *    grd3d_calc_xyz.c
 *
 * DESCRIPTION:
 *    Get X Y Z vector per cell
 *
 * ARGUMENTS:
 *     nx..nz           grid dimensions
 *     coordsv        Coordinates
 *     zcornsv        ZCORN array (pointer) of input
 *     coordsv
 *     p_x_v .. p_z_v   Return arrays for X Y Z
 *     option           0: use all cells, 1: make undef if ACTNUM=0
 *     debug            debug/verbose flag
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
#include "logger.h"

void
grd3d_calc_xyz(int nx,
               int ny,
               int nz,
               double *coordsv,
               long ncoord,
               double *zcornsv,
               long nzcorn,
               int *actnumsv,
               long nact,
               double *p_x_v,
               long npx,
               double *p_y_v,
               long npy,
               double *p_z_v,
               long npz,
               int option)

{

    long ntotv[4] = { nact, npx, npy, npz };
    if (x_verify_vectorlengths(nx, ny, nz, ncoord, nzcorn, ntotv, 4) != 0) {
        throw_exception("Errors in array lengths checks in grd3d_calc_xyz");
        return;
    }
    int i, j, k;
    double xv, yv, zv;

    for (k = 1; k <= nz; k++) {
        for (j = 1; j <= ny; j++) {
            for (i = 1; i <= nx; i++) {

                long ib = x_ijk2ib(i, j, k, nx, ny, nz, 0);
                long ic = x_ijk2ic(i, j, k, nx, ny, nz, 0);

                grd3d_midpoint(i, j, k, nx, ny, nz, coordsv, ncoord, zcornsv, nzcorn,
                               &xv, &yv, &zv);

                p_x_v[ic] = xv;
                p_y_v[ic] = yv;
                p_z_v[ic] = zv;

                if (option == 1 && actnumsv[ib] == 0) {
                    p_x_v[ic] = UNDEF;
                    p_y_v[ic] = UNDEF;
                    p_z_v[ic] = UNDEF;
                }
            }
        }
    }
}
