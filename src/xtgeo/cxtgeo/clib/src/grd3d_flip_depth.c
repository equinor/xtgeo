/*
 * ############################################################################
 * grd3d_flip_depth.c
 * ############################################################################
 */

#include "libxtg.h"
#include "libxtg_.h"

/*
 * ############################################################################
 * Flipping (negate) the Z values of a grid. The parameters shall not be
 * disturbed, as far as I can see.
 * ############################################################################
 */

void grd3d_flip_depth (
		       int   nx,
		       int   ny,
		       int   nz,
		       double *p_coord_v,
		       double *p_zcorn_v,
		       int   debug
		       )

{
    /* locals */
    int    i, j, ic, ib, nzcorn;
    char sub[24]="grd3d_flip_depth";

    xtgverbose(debug);

    xtg_speak(sub,2,"Entering routine ...");

    // coord section

    ib=0;
    for (j=0;j<=ny; j++) {
	for (i=0;i<=nx; i++) {
	    p_coord_v[ib+0] = p_coord_v[ib+0];
	    p_coord_v[ib+1] = p_coord_v[ib+1];
	    p_coord_v[ib+2] = -1 * p_coord_v[ib+2];
	    p_coord_v[ib+3] = p_coord_v[ib+3];
	    p_coord_v[ib+4] = p_coord_v[ib+4];
	    p_coord_v[ib+5] = -1 * p_coord_v[ib+5];
	ib=ib+6;
	}
    }

    // zcorn section
    
    nzcorn=4*nx*ny*(nz+1);
    for (ic=0; ic<=nzcorn; ic++) {
	p_zcorn_v[ic]=-1*p_zcorn_v[ic];
    }
	
    
    xtg_speak(sub,2,"Exit from flipping routine");
}



