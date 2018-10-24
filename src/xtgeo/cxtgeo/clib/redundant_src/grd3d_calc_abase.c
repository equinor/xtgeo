/*
 * ############################################################################
 * grd3d_calc_abase.c
 * Calculates the abase parameter. A computation of dz prior is needed.
 * Author: JCR
 * ############################################################################
 * $Id: grd3d_calc_abase.c,v 1.1 2001/03/14 08:02:29 bg54276 Exp $ 
 * $Source: /d/proj/bg/avresmod/src/gplib/GPLExt/RCS/grd3d_calc_abase.c,v $ 
 *
 * $Log: grd3d_calc_abase.c,v $
 *
 * ############################################################################
 */


#include "libxtg.h"
#include "libxtg_.h"


void grd3d_calc_abase(
		      int option,
		      int nx,
		      int ny,
		      int nz,
		      double *p_dz_v,
		      double *p_abase_v,
		      int flip,
		      int debug
		      )

{
    /* locals */
    int i, j, k, ib, ip;
    char s[24]="grd3d_calc_abase";

    xtgverbose(debug);

    xtg_speak(s,2,"Entering routine");
    xtg_speak(s,3,"NX NY NZ: %d %d %d", nx, ny, nz);


    xtg_speak(s,2,"Finding grid aBase parameter...");

    for (j = 1; j <= ny; j++) {
	for (i = 1; i <= nx; i++) {
	    for (k = nz; k >= 1; k--) {

		ib=x_ijk2ib(i,j,k,nx,ny,nz,0);
		if (k==nz) {
		    p_abase_v[ib]=0.5*p_dz_v[ib];
		}
		else{
		    ip=x_ijk2ib(i,j,k+1,nx,ny,nz,0);		    
		    p_abase_v[ib]=p_abase_v[ip]+0.5*p_dz_v[ib]+0.5*p_dz_v[ip];
		}
	    } 				   
	}
    }
    xtg_speak(s,2,"Exiting <grd3d_calc_abase>");
}
