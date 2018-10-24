/*
 * ############################################################################
 * grd3d_diff_zcorn.c
 * ############################################################################
 */


#include "libxtg.h"
#include "libxtg_.h"

/*
 * ############################################################################
 * Find difference in ZCORN between to grids, and assign them grid propery
 * value
 * ############################################################################
 */

void grd3d_diff_zcorn (
		       int nx,
		       int ny,
		       int nz,
		       double *p_coord1_v,
		       double *p_zcorn1_v,
		       int   *p_actnum1_v,
		       double *p_dz1_v,
		       double *p_dz2_v,
		       double *p_dz3_v,
		       double *p_dz4_v,
		       double *p_dz5_v,
		       double *p_dz6_v,
		       double *p_dz7_v,
		       double *p_dz8_v,
		       double *p_dzavg_v,
		       double *p_coord2_v,
		       double *p_zcorn2_v,
		       int   *p_actnum2_v,
		       int   debug
		       )

{
    /* locals */
    int i, j, k, ib;
    double corner1_v[24], corner2_v[24];
    char s[24]="grd3d_diff_zcorn";

    xtgverbose(debug);

    xtg_speak(s,1,"Entering routine ...");

    for (j = 1; j <= ny; j++) {

	xtg_speak(s,2,"Finished column %d of %d",j,ny);

	for (i = 1; i <= nx; i++) {

	    for (k=1;k<=nz;k++) {
		ib=x_ijk2ib(i,j,k,nx,ny,nz,0);
		if (p_actnum1_v[ib]==1) {
		    /* get the 24 corners for each cell, both grids */
		    grd3d_corners(i,j,k,nx,ny,nz,p_coord1_v,p_zcorn1_v,corner1_v,debug);
		    grd3d_corners(i,j,k,nx,ny,nz,p_coord2_v,p_zcorn2_v,corner2_v,debug);
		    /* compute difference */
		    p_dz1_v[ib]   = corner1_v[2]-corner2_v[2];
		    p_dz2_v[ib]   = corner1_v[5]-corner2_v[5];
		    p_dz3_v[ib]   = corner1_v[8]-corner2_v[8];
		    p_dz4_v[ib]   = corner1_v[11]-corner2_v[11];
		    p_dz5_v[ib]   = corner1_v[14]-corner2_v[14];
		    p_dz6_v[ib]   = corner1_v[17]-corner2_v[17];
		    p_dz7_v[ib]   = corner1_v[20]-corner2_v[20];
		    p_dz8_v[ib]   = corner1_v[23]-corner2_v[23];
		    p_dzavg_v[ib] = 0.125*(p_dz1_v[ib]+p_dz2_v[ib]+p_dz3_v[ib]+p_dz4_v[ib]+
					   p_dz5_v[ib]+p_dz6_v[ib]+p_dz7_v[ib]+p_dz8_v[ib]);
		}
	    }
	}
    }

    xtg_speak(s,1,"Exiting ...");
}
