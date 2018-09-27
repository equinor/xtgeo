/*
 * ############################################################################
 * grd3d_merge_grids.c
 * Will merge (stack) two grids with same COORDs. The added layer will have
 * ACTNUM 0.
 * Arguments:
 * nx ny          NX and NY for both grids (must be the same)
 * nz1            NZ for grid no 1
 * p_coord1_v     Coordinates for grid no 1; will be used for result! (actually
 *                this goes not into the current calculations...)
 * p_zcorn1_v     ZCORNs for grid no 1
 * p_actnum1_v    ACTNUM for grid no 1
 * p_zcorn2_v     ZCORNs for grid no 2
 * p_actnum2_v    ACTNUM for grid no 2
 * nznew          new NZ (nz1+nz2+1)
 * p_zcornnew_v   New ZCORN;
 * p_actnumnew_v  New ZCORN;
 * ############################################################################
 */


#include "libxtg.h"
#include "libxtg_.h"

/*
 * ############################################################################
 * ############################################################################
 */

void grd3d_merge_grids (
			int   nx,
			int   ny,
			int   nz1,
			double *p_coord1_v,
			double *p_zcorn1_v,
			int   *p_actnum1_v,
			int   nz2,
			double *p_zcorn2_v,
			int   *p_actnum2_v,
			int   nznew,
			double *p_zcornnew_v,
			int   *p_actnumnew_v,
			int   *p_numnew_act,
			int   option,
			int   debug
			   )
    
{
    /* locals */
    int    i, j, k, knew, ic, mi, ib1, ib2, ibn,  actcount;
    char s[24]="grd3d_merge_grids";

    xtgverbose(debug);

    xtg_speak(s,1,"Entering <grd3d_merge_grids>");

    actcount=0;

    for (j = 1; j <= ny; j++) {
	mi=j % 10; /* modulus */
 	if (mi==0) xtg_speak(s,1,"Finished column %d of %d",j,ny);
	if (mi!=0 && j==ny) xtg_speak(s,1,"Finished column %d of %d",j,ny);

	for (i = 1; i <= nx; i++) {

	    for (k = 1; k <= nz1+1; k++) {

		ib1 = x_ijk2ib(i,j,k,nx,ny,nz1+1,0);
		ibn = x_ijk2ib(i,j,k,nx,ny,nznew+1,0);

		/* do for all corners */
		for (ic=1;ic<=4;ic++) {
		   p_zcornnew_v[4*ibn + 1*ic - 1] = p_zcorn1_v[4*ib1 + 1*ic - 1];
		}

		/* ACTNUM */
		if (k<=nz1) {
		    actcount += p_actnum1_v[ib1];
		    p_actnumnew_v[ibn]=p_actnum1_v[ib1];
		}
		else{
		    /* the cell layer between the two grids */
		    p_actnumnew_v[ibn]=option;
		}		    

	    }

	    /* add the other grid */
	    for (k = 1; k <= nz2+1; k++) {
		knew=nz1+1+k;

		ib2 = x_ijk2ib(i,j,k,    nx,ny,nz2+1,0);
		ibn = x_ijk2ib(i,j,knew, nx,ny,nznew+1,0);

		/* do for all corners */
		for (ic=1;ic<=4;ic++) {
		   p_zcornnew_v[4*ibn + 1*ic - 1] = p_zcorn2_v[4*ib2 + 1*ic - 1];
		}

		/* ACTNUM */
		if (k<=nz2) {
		    actcount += p_actnum2_v[ib2];
		    p_actnumnew_v[ibn]=p_actnum2_v[ib2];
		}
	    }
	
	}
	
    }

    /* update number of active cells */
    *p_numnew_act = actcount;

    xtg_speak(s,2,"Exit from <%s>",s);

}

