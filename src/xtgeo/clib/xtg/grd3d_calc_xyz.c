/*
 * #############################################################################
 * Name:      grd3d_calc_xyz.c
 * Author:    jriv@statoil.com
 * #############################################################################
 * Get X Y Z vectors per cell
 *
 *
 * Return:
 *     void function, returns updated pointers to arrays
 *
 * Caveeats/issues:
 *
 * #############################################################################
 */
/*
****************************************************************************************
 *
 * NAME:
 *    grd3d_calc_xyz.c
 *
 * DESCRIPTION:
 *    Get X Y Z vector per cell
 *
 * ARGUMENTS:
 *     nx..nz           grid dimensions
 *     p_coord_v        Coordinates
 *     p_zcorn_v        ZCORN array (pointer) of input
 *     p_coord_v
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

#include "logger.h"
#include "libxtg.h"
#include "libxtg_.h"


void grd3d_calc_xyz(
    int nx,
    int ny,
    int nz,
    double *p_coord_v,
    long ncoord,
    double *p_zcorn_v,
    long nzcorn,
    int *p_actnum_v,
    long nact,
    double *p_x_v,
    long npx,
    double *p_y_v,
    long npy,
    double *p_z_v,
    long npz,
    int option
    )

{
    /* locals */
    int     i, j, k;
    double  xv, yv, zv;

    for (k = 1; k <= nz; k++) {
	for (j = 1; j <= ny; j++) {
	    for (i = 1; i <= nx; i++) {

		long ib = x_ijk2ib(i, j, k, nx, ny, nz, 0);
		long ic = x_ijk2ic(i, j, k, nx, ny, nz, 0);

		grd3d_midpoint(i, j, k, nx, ny, nz, p_coord_v, p_zcorn_v,
			       &xv, &yv, &zv, XTGDEBUG);


		p_x_v[ic] = xv;
		p_y_v[ic] = yv;
		p_z_v[ic] = zv;

		if (option==1 && p_actnum_v[ib]==0) {
		    p_x_v[ic] = UNDEF;
		    p_y_v[ic] = UNDEF;
		    p_z_v[ic] = UNDEF;
		}

	    }
	}
    }

}
