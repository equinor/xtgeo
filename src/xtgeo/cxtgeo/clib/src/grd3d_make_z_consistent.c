/*
 * ############################################################################
 * grd3d_make_z_consistent.c
 * ############################################################################
 * $Id: grd3d_adj_z_from_map.c,v 1.3 2001/03/14 08:02:29 bg54276 Exp bg54276 $ 
 * $Source: /d/proj/bg/avresmod/src/gplib/GPLExt/RCS/grd3d_adj_z_from_map.c,v $ 
 *
 * $Log: grd3d_adj_z_from_map.c,v $
 *
 * ############################################################################
 */


#include "libxtg.h"
#include "libxtg_.h"

/*
 * ############################################################################
 * Some other functions will fail if z-nodes are inconsistent in depth (e.g.
 * grd3_adj_z_from_map). This routine will make z consistent, and also add
 * a (very) small separation (zsep) if the separation is too small
 * ############################################################################
 */

void grd3d_make_z_consistent (
			      int    nx,
			      int    ny,
			      int    nz,
			      double *p_zcorn_v,
			      int    *p_actnum_v,
			      double  zsep,
			      int    debug
			      )

{
    /* locals */
    int    i, j, k, ic, ibp, ibx;
    double  z1, z2;
    char sub[24]="grd3d_make_z_consistent";

    xtgverbose(debug);

    xtg_speak(sub,2,"Entering <grd3d_make_z_consistent>");
    xtg_speak(sub,3,"Minimum cell Z seperation is %f", zsep);
    
    for (j = 1; j <= ny; j++) {
	xtg_speak(sub,4,"Finished column %d of %d",j,ny);
	for (i = 1; i <= nx; i++) {
	    for (k = 2; k <= nz+1; k++) {
		ibp=x_ijk2ib(i,j,k-1,nx,ny,nz+1,0);
		ibx=x_ijk2ib(i,j,k,nx,ny,nz+1,0);
		for (ic=1;ic<=4;ic++) {
		    z1=p_zcorn_v[4*ibp + 1*ic - 1];
		    z2=p_zcorn_v[4*ibx + 1*ic - 1];
		    /* if ((z2-z1) < zsep && p_actnum_v[ibp]==1) { */
		    if ((z2-z1) < zsep) { 
			if (debug > 3) {
			    xtg_warn(sub,4,"Too small dZ found at I=%d J=%d K=%d",
				     i,j,k);
			    xtg_warn(sub,4,"Corner %d Ztop=%f and Zbot=%f",
				     ic,z1,z2);
			    
			    if (z2<z1) { /* this will occur due to recursion! */
				xtg_warn(sub,4,"Negative dZ found at I=%d J=%d K=%d",
					 i,j,k);
				xtg_warn(sub,4,"Corner %d Ztop=%f and Zbot=%f",
					 ic,z1,z2);
				
			    }
			}
			p_zcorn_v[4*ibx + 1*ic - 1] = z1 + zsep;
		    }
		}
	    }
	}
    }
    
    xtg_speak(sub,2,"Exiting <grd3d_make_z_consistent>");
}



