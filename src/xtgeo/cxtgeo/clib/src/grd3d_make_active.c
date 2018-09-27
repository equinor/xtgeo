/*
 * ############################################################################
 * grd3d_make_active.c
 * Make all cells active
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
 *                          GRD3D_MAKE_ACTIVE
 * Make grid cells active. No particular checks, yet ...
 * ****************************************************************************
 * ----------------------------------------------------------------------------
 *
 */   
void grd3d_make_active(
		       int     i1,
		       int     i2,
		       int     j1,
		       int     j2,
		       int     k1,
		       int     k2,
		       int     nx,
		       int     ny,
		       int     nz,
		       int     *p_actnum_v,
		       int     debug
		       )
{
    int i, j, k, ib, no;
    char s[24]="grd3d_make_active";

    xtgverbose(debug);

    xtg_speak(s,2,"Make cells active ...");

    no=0;
    for (k=k1;k<=k2;k++) {
	for (j=j1;j<=j2;j++) {
	    for (i=i1;i<=i2;i++) {
	      ib=x_ijk2ib(i,j,k,nx,ny,nz,0);
	      p_actnum_v[ib]=1;
	    }
	}
    }
}


