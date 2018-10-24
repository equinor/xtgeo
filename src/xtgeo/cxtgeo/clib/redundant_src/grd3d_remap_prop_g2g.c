/*
 * ############################################################################
 * grd3d_remap_prop_g2g.c
 * Sample values from one grid into another. If g2 is a subset (subgrid) of
 * grid 1, then grid 1 can get a copy of those values in g2
 * Author: JCR
 *
 * Todo: Checking that bounds are corresponding
 * ############################################################################
 * $Id: grd3d_remap_prop_g2g.c,v 1.2 2001/03/14 08:02:29 bg54276 Exp $ 
 * $Source: /d/proj/bg/avresmod/src/gplib/GPLExt/RCS/grd3d_remap_prop_g2g.c,v $ 
 *
 * $Log: grd3d_remap_prop_g2g.c,v $
 * Revision 1.2  2001/03/14 08:02:29  bg54276
 * *** empty log message ***
 *
 * Revision 1.1  2000/12/12 17:24:54  bg54276
 * Initial revision
 *
 *
 * ############################################################################
 */


#include "libxtg.h"
#include "libxtg_.h"


void grd3d_remap_prop_g2g(
			   int    nx1,
			   int    ny1,
			   int    nz1,
			   int    nx2,
			   int    ny2,
			   int    nz2,
			   int    isub,
			   int    num_subgrds,
			   int    *p_subgrd_v,
			   char   *ptype,
			   int    *p_int1_v,
			   double *p_dfloat1_v,
			   int    *p_int2_v,
			   double *p_dfloat2_v,
			   int    debug
			   )

{
    /* locals */
    int i, j, k, m, ib1, ib2, nz1a, nz1b, kc;
    char s[24]="grd3d_remap_prop_g2g";

    xtgverbose(debug);
    xtg_speak(s,2,"Entering <grd3d_remap_prop_g2g>");
    xtg_speak(s,3,"NX1 NY1 NZ1: %d %d %d", nx1, ny1, nz1);
    xtg_speak(s,3,"NX2 NY2 NZ2: %d %d %d", nx2, ny2, nz2);

    nz1a=1;
    nz1b=nz1;

    if (isub <= num_subgrds) {
	if (isub > 0) {
	    
	    /* find nz1 and nz2 (counted from top) */
	    k=0;
	    for (kc=0;kc<(isub-1);kc++) {
		k=k+p_subgrd_v[kc];
	    }
	    nz1a=k+1;
	    nz1b=k+p_subgrd_v[isub-1];
	    xtg_speak(s,2,"Subgrid, using K range: %d - %d",nz1a,nz1b);
	}
    }
    else{
	xtg_error(s,"Fatal error: isub too large");
    }



    xtg_speak(s,2,"Remapping...");

    m=0;
    for (k = nz1a; k <= nz1b; k++) {
	m++;
	xtg_speak(s,3,"Finished layer %d(%d) of %d(%d)",k,m,nz1,nz2);
	for (j = 1; j <= ny1; j++) {
	    for (i = 1; i <= nx1; i++) {
		ib1=x_ijk2ib(i,j,k,nx1,ny1,nz1,0);
		ib2=x_ijk2ib(i,j,m,nx2,ny2,nz2,0);

		if (strcmp(ptype,"double")==0) {		    
		    p_dfloat1_v[ib1]=p_dfloat2_v[ib2];
		}
		else{
		    p_int1_v[ib1]=p_int2_v[ib2];
		}			    
	    } 				   
	}
    }
    xtg_speak(s,2,"Remapping... DONE!");

    xtg_speak(s,2,"Exiting <grd3d_remap_prop_g2g>");
}
