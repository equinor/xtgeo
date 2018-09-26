/*
 * ############################################################################
 * grd3d_match_zonebounds.c
 * ############################################################################
 * $Id: grd3d_adj_z_from_map.c,v 1.3 2001/03/14 08:02:29 bg54276 Exp bg54276 $
 * $Source: /d/proj/bg/avresmod/src/gplib/GPLExt/RCS/grd3d_adj_z_from_map.c,v $
 *
 * $Log: grd3d_adj_z_from_map.c,v $
 * Revision 1.3  2001/03/14 08:02:29  bg54276
 * *** empty log message ***
 *
 * Revision 1.1  2001/01/01 20:03:38  bg54276
 * Initial revision
 *
 *
 *
 * ############################################################################
 */


#include "libxtg.h"
#include "libxtg_.h"
#define  ZTOLERANCE 0.1
/*
 * ############################################################################
 * Ensure that zone boundaries are equal in Z
 * ############################################################################
 */

void grd3d_match_zonebounds (
			     int nx,
			     int ny,
			     int nz1,
			     int nz2,
			     double *p_zcorn1_v,
			     int   *p_actnum1_v,
			     double *p_zcorn2_v,
			     int   *p_actnum2_v,
			     int   iflag,
			     int   debug
			   )

{
    /* locals */
    int i, j, k, k1, k2, ic, ib;
    double c1z, c2z;
    char s[24] = "grd3d_match_z..bounds";

    xtgverbose(debug);

    xtg_speak(s, 1,"Entering <grd3d_match_zonebounds>");
    xtg_speak(s, 3,"Using IFLAG: %d", iflag);

    xtg_speak(s, 3,"NX NY NZ1: %d %d %d", nx, ny, nz1);
    xtg_speak(s, 3,"NX NY NZ2: %d %d %d", nx, ny, nz2);

    xtg_speak(s, 1,"Matching zone boundaries - this may take some time...");
    for (j = 1; j <= ny; j++) {
	xtg_speak(s, 2,"Finished column %d of %d",j,ny);
	for (i = 1; i <= nx; i++) {

	    /* bottom Z of upper grid */
	    k1=0;
	    k2=0;
	    for (k=nz1;k>=1;k--) {
		ib=x_ijk2ib(i,j,k,nx,ny,nz1+1,0);
		if (p_actnum1_v[ib]==1) {
		    k1=k;
		    break;
		}
	    }

	    for (k=1;k<=nz2;k++) {
		ib=x_ijk2ib(i,j,k,nx,ny,nz2+1,0);
		if (p_actnum2_v[ib]==1) {
		    k2=k;
		    break;
		}
	    }


	    /* The full cell column may be inactive */
	    if (k1==0 || k2==0) goto NEXTLOOP;

	    for (ic=1;ic<=4;ic++) {

		ib=x_ijk2ib(i,j,k1+1,nx,ny,nz1+1,0);
		c1z=p_zcorn1_v[4*ib + 1*ic - 1];

		ib=x_ijk2ib(i,j,k2,nx,ny,nz2+1,0);
		c2z=p_zcorn2_v[4*ib + 1*ic - 1];

		/* if there is a gap, just move the grid2 uppermost cell to match */
		if (c1z < (c2z-ZTOLERANCE)) {
		    p_zcorn2_v[4*ib + 1*ic - 1]=c1z;
		}
		/* if there is overlap, just move the grid2 uppermost cell to match, and make consistent! */
		else if (c1z > (c2z+ZTOLERANCE)) {
		    p_zcorn2_v[4*ib + 1*ic - 1]=c1z;
		}
	    }
	NEXTLOOP:
	    xtg_speak(s, 4,"Skipping column...");
	}
    }

    grd3d_make_z_consistent(
			    nx,
			    ny,
			    nz2,
			    p_zcorn2_v,
			    p_actnum2_v,
			    0.0001,
			    debug
			    );

    xtg_speak(s, 2,"Matching zone bounds ... DONE!");
    xtg_speak(s, 1,"Exiting <grd3d_match_zonebounds>");
}
