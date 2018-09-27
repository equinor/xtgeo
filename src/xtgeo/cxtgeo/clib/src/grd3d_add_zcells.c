/*
 * #############################################################################
 * grd3d_add_zcells.c
 * Will add (stack) cells in K with same COORDs. The added layers will have
 * ACTNUM 1.
 * Arguments:
 * nx ny          NX and NY for this grids (will be unchagned)
 * nz             NZ for input
 * nadd           number of cells added
 * thickness      total thickness of added cells
 * p_zcorn1_v     ZCORNs for grid input
 * p_actnum1_v    ACTNUM for grid input
 * nznew          new NZ (nz1+nz)
 * p_zcornnew_v   New ZCORN;
 * p_actnumnew_v  New ACTNUM;
 * option         0: add at top, 1 add at bottom
 *
 * Note:
 * Coordinates (pillars) are not applied (will just inherit existent)
 * Author:
 * Jan C. Rivenaes
 * #############################################################################
 */


#include "libxtg.h"
#include "libxtg_.h"

/*
 * #############################################################################
 */

void grd3d_add_zcells (
		       int    nx,
		       int    ny,
		       int    nz,
		       int    nadd,
		       double thickness,
		       double *p_zcorn1_v,
		       int    *p_actnum1_v,
		       double *p_zcornnew_v,
		       int    *p_actnumnew_v,
		       int    *p_numnew_act,
		       int    option,
		       int    debug
		       )

{
    /* locals */
    int    i, j, k, knew, ic, mi, ib1, ib2, ibn,  actcount, nznew;
    char   s[24]="grd3d_add_zcells";
    double dz;

    xtgverbose(debug);

    xtg_speak(s,1,"Entering <grd3d_merge_grids>");

    actcount = 0;
    
    nznew = nz + nadd;

    dz=thickness/nadd;

    if (option==0) {
       	xtg_error(s,"option = 0 ... not implemented yet");
    }

    
    xtg_speak(s,1,"Add %3d layers to the base, with total thickness %7.3f", 
	      nadd, thickness);

    for (j = 1; j <= ny; j++) {
	mi=j % 10; /* modulus */
 	if (mi==0) xtg_speak(s,1,"Finished column %d of %d",j,ny);
	if (mi!=0 && j==ny) xtg_speak(s,1,"Finished column %d of %d",j,ny);

	for (i = 1; i <= nx; i++) {

	    for (k = 1; k <= nz+1; k++) {

		ib1 = x_ijk2ib(i,j,k,nx,ny,nz+1,0);
		ibn = x_ijk2ib(i,j,k,nx,ny,nznew+1,0);

		/* do for all corners */
		for (ic=1;ic<=4;ic++) {
		   p_zcornnew_v[4*ibn + 1*ic - 1] = p_zcorn1_v[4*ib1 + 1*ic - 1];
		}

		/* ACTNUM */
		if (k<=nz) {
		    ib2 = x_ijk2ib(i,j,k,nx,ny,nz,0);

		    
		    if (debug >2 && i==47 && j==69 && k==13) {
			xtg_speak(s,1,"IB1, IB2, IBN %d %d %d : %d", 
				  ib1, ib2, ibn, p_actnum1_v[ib2]);
		    }
		    
		    actcount += p_actnum1_v[ib2];
		    p_actnumnew_v[ib2]=p_actnum1_v[ib2];

		    if (ib1 != ibn || ib2 != ibn) {
			xtg_speak(s,1,"Something is rotten: IB1 vs IBN: %d vs %d",ib1,ibn);
		    }	
		    
		}
	    }

	    /* add the extra layers */
	    for (k = 1; k <= nadd; k++) {
		knew=nz+1+k;
		
		ib1 = x_ijk2ib(i,j,nz+1, nx,ny,nz+1,0);
		ibn = x_ijk2ib(i,j,knew, nx,ny,nznew+1,0);

		/* do for all corners */
		for (ic=1;ic<=4;ic++) {
		   p_zcornnew_v[4*ibn + 1*ic - 1] = p_zcorn1_v[4*ib1 + 1*ic - 1] + k*dz;
		}

		/* ACTNUM */
		ib2 = x_ijk2ib(i,j,knew-1, nx,ny,nznew,0);
		actcount += 1;
		p_actnumnew_v[ib2]=1;
	    }
	
	}
	
    }

    /* update number of active cells */
    *p_numnew_act = actcount;

    xtg_speak(s,2,"Exit from <%s>",s);

}

