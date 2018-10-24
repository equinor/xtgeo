/*
 * ############################################################################
 * grd3d_count_active.c
 * Counts the number of active cells; return it with function
 * Author: J.C. Rivenaes
 * ############################################################################
 * $Id: $ 
 * $Source: $ 
 *
 * $Log: $
 *
 * ############################################################################
 * General description:
 * ############################################################################
 */

#include "libxtg.h"
#include "libxtg_.h"

/*
 * ****************************************************************************
 *                          GRD2D_POLYGON_MASK
 * ****************************************************************************
 * The algorithm is to see if the map nodes lies inside some of the polygins.
 * If not, an undef value is given. If already undef, then value is kept.
 * Todo: The algorithm is straightforward and hence a bit slow...
 * ----------------------------------------------------------------------------
 *
 */   
int grd3d_count_active(
		       int     nx,
		       int     ny,
		       int     nz,
		       int    *p_actnum_v,
		       int debug
		       )
{
    int i, j, k, ib, no;
    char s[24]="grd3d_count_inactive";

    xtgverbose(debug);

    xtg_speak(s,2,"Count active cells ...");

    no=0;
    for (k=1;k<=nz;k++) {
	for (j=1;j<=ny;j++) {
	    for (i=1;i<=nx;i++) {
	      ib=x_ijk2ib(i,j,k,nx,ny,nz,0);
	      no += p_actnum_v[ib];
	    }
	}
    }
    return no;
}


