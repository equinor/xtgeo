/*
 * ############################################################################
 * grd3d_diff_zcorn.c
 * ############################################################################
 */


#include "libxtg.h"
#include "libxtg_.h"

/*
 * ############################################################################
 * Adjust the grid by using a delta ZCORN parameter for each corner
 * Note that the definition in XTGeo is pr layer, so an 8 cornerpoint may
 * specify a conflict, e.g. dZ5(iz) must be equal to dZ1(iz+1).
 * ############################################################################
 */

void grd3d_adj_dzcorn (
		       int nx,
		       int ny,
		       int nz,
		       double *p_zcorn_v,
		       double *p_dz1_v,
		       double *p_dz2_v,
		       double *p_dz3_v,
		       double *p_dz4_v,
		       double *p_dz5_v,
		       double *p_dz6_v,
		       double *p_dz7_v,
		       double *p_dz8_v,
		       int   debug
		       )

{
    /* locals */
    int i, j, k, ibt, ibb;
    char s[24]="grd3d_adj_dzcorn";

    xtgverbose(debug);

    xtg_speak(s,1,"Entering routine ...");

    for (j = 1; j <= ny; j++) {

	xtg_speak(s,2,"Finished column %d of %d",j,ny);

	for (i = 1; i <= nx; i++) {

	    for (k=1;k<=nz;k++) {
		ibt=x_ijk2ib(i,j,k,nx,ny,nz,0);    /* top and midcell count*/
		ibb=x_ijk2ib(i,j,k+1,nx,ny,nz,0);  /* bottom cell surface */

		/* assumming that bottom corner adjustment is equal to
		   top of next cell below */

		p_zcorn_v[4*ibt + 1*1 - 1] =
                    p_zcorn_v[4*ibt + 1*1 - 1] - p_dz1_v[ibt];

		p_zcorn_v[4*ibt + 1*2 - 1] =
                    p_zcorn_v[4*ibt + 1*2 - 1] - p_dz2_v[ibt];

		p_zcorn_v[4*ibt + 1*3 - 1] =
                    p_zcorn_v[4*ibt + 1*3 - 1] - p_dz3_v[ibt];

		p_zcorn_v[4*ibt + 1*4 - 1] =
                    p_zcorn_v[4*ibt + 1*4 - 1] - p_dz4_v[ibt];

		if (k==nz) {
		    p_zcorn_v[4*ibb + 1*1 - 1] =
                        p_zcorn_v[4*ibb + 1*1 - 1] - p_dz5_v[ibt];

		    p_zcorn_v[4*ibb + 1*2 - 1] =
                        p_zcorn_v[4*ibb + 1*2 - 1] - p_dz6_v[ibt];

		    p_zcorn_v[4*ibb + 1*3 - 1] =
                        p_zcorn_v[4*ibb + 1*3 - 1] - p_dz7_v[ibt];

		    p_zcorn_v[4*ibb + 1*4 - 1] =
                        p_zcorn_v[4*ibb + 1*4 - 1] - p_dz8_v[ibt];
		}

	    }
	}
    }

    /* need consistensy test ?? */


    xtg_speak(s,1,"Exiting ...");
}
