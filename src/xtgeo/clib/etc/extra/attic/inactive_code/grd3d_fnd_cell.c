/*
 * ############################################################################
 * Find a cell given a XYZ point. The search is NOT smart, it just loops the
 * whole shit...
 * The routine return -1 if no cell is found 
* Author: JCR
 * ############################################################################
 * $Id: $ 
 * $Source: $ 
 *
 * $Log: $
 * ############################################################################
 */


#include <string.h>
#include <stdlib.h>
#include "libxtg.h"
#include "libxtg_.h"
/*
 * i,j,k       Output i,j,k (input could be anything)
 * x,y,z       Point to be checked if inside cell
 */

int _grd3d_fnd_cell (
		     int     *i,
		     int     *j,
		     int     *k,
		     int     nx,
		     int     ny,
		     int     nz,
		     float   *p_coord_v,
		     float   *p_zgrd3d_v,
		     int     *p_actnum_v,
		     float   x,
		     float   y,
		     float   z,
		     int     debug
		    )


{
    char  s[24]="grd3d_fnd_cell";
    int   ir, ix, jy, kz, ii, jj, kk, ia, ib, ok;
    float corners[24];

    xtgverbose(debug);

    xtg_speak(s,4,"==== Entering routine ====");

    /* 
     * The quest for the search is to go from I,J,K
     * and seek out as a "square blowing balloon"
     */
    
    ix = *i; jy = *j; kz = *k;
    for (ii=1; ii<=nx; ii++) {
      for (jj=1; jj<=ny; jj++) {
	for (kk=1; kk<=nz; kk++) {
      
	  /* get the cell corners */
	  grd3d_corners(ii,jj,kk,nx,ny,nz,
			p_coord_v, p_zgrd3d_v,
			corners, debug);
	  
	  
	  /* check if point is inside cell */
	  
	  ib=x_ijk2ib(ii,jj,kk,nx,ny,nz,0);
	  ia=p_actnum_v[ib];	    
	  ok=x_chk_point_in_cell(x,y,z,corners,1,debug);
	  
	  if (ok>0) {
	      xtg_speak(s,3,"Cell found %d %d %d", ii, jj, kk);
	    *i=ii; *j=jj; *k=kk;
	    return ia; /* 1 if active, 0 if inactive */
	  }
	}
      }
    }

    xtg_speak(s,4,"==== Exiting routine ====");


    return -1;
}


