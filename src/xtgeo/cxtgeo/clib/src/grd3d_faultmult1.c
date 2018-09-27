/*
 * ############################################################################
 * grd3d_faultmult1.c
 * Computes fault multipliers based on SGR theory
 * Input:
 *          nx
 *          ny
 *          nz
 *          p_coord_v
 *          pzcorn_v
 *          p_actnum_v
 *          p_fmark_v
 *
 * Output:
 *          p_multx_v   = array (nx*ny*nz) that holds upstream multx of faults
 *                        (must be allocated and set to 1 in calling routine)
 *          p_multy_v   = array (nx*ny*nz) that holds upstream multx of faults
 *                        (must be allocated and set to 1 in calling routine)
 *
 * Author:  JCR
 * ############################################################################
 * $Id: grd3d_faultmult1.c,v 1.1 2001/09/28 20:18:36 bg54276 Exp $
 * $Source: /d/proj/bg/avresmod/src/gplib/GPLExt/RCS/grd3d_faultmult1.c,v $
 *
 * $Log: grd3d_faultmult1.c,v $
 * Revision 1.1  2001/09/28 20:18:36  bg54276
 * Initial revision
 *
 *
 * ############################################################################
 */


#include <math.h>
#include "libxtg.h"
#include "libxtg_.h"


void grd3d_faultmult1(
		      int nx,
		      int ny,
		      int nz,
		      double *p_coord_v,
		      double *p_zcorn_v,
		      int   *p_actnum_v,
		      int   *p_fmark_v,
		      double flimit,
		      int   debug
		      )

{
    /* locals */

    int i, j, k, ib, in, ip, iq;
    char s[24] = "grd3d_faultmult1";

    xtgverbose(debug);

    xtg_speak(s,1,"Entering %s", s);
    xtg_speak(s,3,"NX NY NZ: %d %d %d", nx, ny, nz);
    xtg_speak(s,1,"Fault limit is %f", flimit);

    /*initiliasing to 0 is done in Perl ptrcreate*/

    xtg_speak(s,1,"Finding faults...");

    for (k = 1; k <= nz; k++) {
	xtg_speak(s,3,"Finished layer %d of %d",k,nz);
	/* loop in X */
	for (j = 1; j <= ny; j++) {
	    for (i = 1; i < nx; i++) {
		/* parameter counting */
		ib=x_ijk2ib(i,j,k,nx,ny,nz,0);
		in=x_ijk2ib(i+1,j,k,nx,ny,nz,0);

		/* grid */
		ip=x_ijk2ib(i,  j,k,nx,ny,nz+1,0);
		iq=x_ijk2ib(i+1,j,k,nx,ny,nz+1,0);

		if (p_actnum_v[ib]==1 && p_actnum_v[in]==1) {
		    /* corner 3/4 for cell ip is faulted to 1/2 for iq */
		    if (fabs(p_zcorn_v[4*ip+2-1]-p_zcorn_v[4*iq+1-1])>flimit ||
			fabs(p_zcorn_v[4*ip+4-1]-p_zcorn_v[4*iq+3-1])>flimit){
			p_fmark_v[ib]=1;
		    }
		}
	    }
	}
	/* loop in Y */
	for (i = 1; i <= nx; i++) {
	    for (j = 1; j < ny; j++) {
		/* parameter counting */
		ib=x_ijk2ib(i,j,k,nx,ny,nz,0);
		in=x_ijk2ib(i,j+1,k,nx,ny,nz,0);

		/* grid */
		ip=x_ijk2ib(i,  j,k,nx,ny,nz+1,0);
		iq=x_ijk2ib(i,j+1,k,nx,ny,nz+1,0);

		if (p_actnum_v[ib]==1 && p_actnum_v[in]==1) {
		    /* corner 2/4 for cell ip is faulted to 1/3 for iq */
		    if (fabs(p_zcorn_v[4*ip+3-1]-p_zcorn_v[4*iq+1-1])>flimit ||
			fabs(p_zcorn_v[4*ip+4-1]-p_zcorn_v[4*iq+2-1])>flimit){
			p_fmark_v[ib]+=2;
		    }
		}

	    }
	}
    }
    xtg_speak(s,1,"Exiting %s", s);
}
