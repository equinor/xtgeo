/*
 * #################################################################################################
 * Name:      grd3d_calc_sum_dz.c
 * Author:    JRIV@statoil.com
 * Created:   2001-03-14
 * Updates:   Minor
 * #################################################################################################
 * Calculates summary DZ for a grid, and put the same value in all cells
 * in column. The summation does not include inactive cells at bottom(??)
 *
 * Arguments:
 *     nx..nz           grid dimensions
 *     p_zcorn_v        ZCORN array (pointer) of input
 *     p_actnum_v       ACTNUM array (pointer)
 *     p_sumdz_v        resulting DZ sum array (pointer)
 *     flip             use 1 or -1 for flipping vertical
 *     debug            debug/verbose flag
 *
 * Caveeats/issues:
 *     nothing known 
 * #################################################################################################
 */


#include "libxtg.h"
#include "libxtg_.h"


void grd3d_calc_sum_dz(
		      int     nx,
		      int     ny,
		      int     nz,
		      double  *p_zcorn_v,
		      int     *p_actnum_v,
		      double  *p_sumdz_v,
		      int     flip,
		      int     debug
		      )

{
    /* locals */
    int       i, j, k, ib;
    double     sum;
    double     *dz_v;
    char      s[24]="grd3d_calc_sum_dz";

    xtgverbose(debug);
    xtg_speak(s,2,"Entering <grd3d_calc_sum_dz>");
    xtg_speak(s,3,"NX NY NZ: %d %d %d", nx, ny, nz);

    dz_v=calloc(nx*ny*nz,sizeof(*dz_v));
    if (dz_v==NULL) {
	xtg_error(s,"STOP! Cannot allocate memory!");
    }

    grd3d_calc_dz(
		  nx,
		  ny,
		  nz,
		  p_zcorn_v,
		  p_actnum_v,
		  dz_v,
		  flip,
		  0,
		  debug
		  );
    
    xtg_speak(s,2,"Finding grid SUM DZ parameter...");

    for (j = 1; j <= ny; j++) {
	for (i = 1; i <= nx; i++) {
	    sum=0.0;
	    for (k = 1; k <= nz; k++) {
		
		/* parameter counting */
		ib=x_ijk2ib(i,j,k,nx,ny,nz,0);
		
		/* the 2 is a special COMPDAT case */
		if (p_actnum_v[ib] >= 1) {
		    sum=sum+dz_v[ib];
		}
	    }
	    
	    /* now put found values in grid */
	    for (k = 1; k <= nz; k++) {
	       /* parameter counting */
		ib=x_ijk2ib(i,j,k,nx,ny,nz,0);
		
		if (p_actnum_v[ib] == 1 || p_actnum_v[ib] == -1) {
		    p_sumdz_v[ib]=sum;
		}
	    }
	}
    }

    free(dz_v);
    xtg_speak(s,2,"Exiting <grd3d_calc_sum_dz>");
}


