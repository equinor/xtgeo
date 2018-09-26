/*
 * #############################################################################
 * Name:      grd3d_calc_xyz.c
 * Author:    jriv@statoil.com
 * #############################################################################
 * Get X Y Z vectors per cell
 *
 * Arguments:
 *     nx..nz           grid dimensions
 *     p_coord_v        Coordinates
 *     p_zcorn_v        ZCORN array (pointer) of input
 *     p_coord_v
 *     p_x_v .. p_z_v   Return arrays for X Y Z
 *     option           0: use all cells, 1: make undef if ACTNUM=0
 *     debug            debug/verbose flag
 *
 * Return:
 *     void function, returns updated pointers to arrays
 *
 * Caveeats/issues:
 *
 * #############################################################################
 */

#include "libxtg.h"
#include "libxtg_.h"


void grd3d_calc_xyz(
		    int      nx,
		    int      ny,
		    int      nz,
		    double   *p_coord_v,
		    double   *p_zcorn_v,
		    int      *p_actnum_v,
		    double   *p_x_v,
		    double   *p_y_v,
		    double   *p_z_v,
		    int      option,
		    int      debug
		    )
    
{
    /* locals */
    int     i, j, k, ib;
    double  xv, yv, zv;
    char    s[24]="grd3d_calc_xyz";

    xtgverbose(debug);

    xtg_speak(s,2,"Entering %s",s);
    xtg_speak(s,3,"NX NY NZ: %d %d %d", nx, ny, nz);

    xtg_speak(s,2,"Finding cell centers...");

    for (k = 1; k <= nz; k++) {
	xtg_speak(s,3,"Finished layer %d of %d",k,nz);
	for (j = 1; j <= ny; j++) {
	    for (i = 1; i <= nx; i++) {

		ib=x_ijk2ib(i,j,k,nx,ny,nz,0);
		
		grd3d_midpoint(i,j,k,nx,ny,nz,p_coord_v,p_zcorn_v,
			       &xv, &yv, &zv, debug);
		
		
		p_x_v[ib]=xv;
		p_y_v[ib]=yv;
		p_z_v[ib]=zv;

		if (option==1 && p_actnum_v[ib]==0) {
		    p_x_v[ib]=UNDEF;
		    p_y_v[ib]=UNDEF;
		    p_z_v[ib]=UNDEF;
		}
		
	    }
	}
    }

    xtg_speak(s,2,"Exit from %s",s);
}
