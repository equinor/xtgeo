/*
 * ############################################################################
 * Find the nearesst cell given a XYZ point and a start IJK cell. The search
 * should be smart to minimuze CPU usage. It will be applied by other routines
 * that tries to find points that comes close after eachother
 * The routine return 0 if no cell is found 
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
 * i,j,k       Input i,j,k and output (modified) i,j,k
 * x,y,z       Point to be checked if inside cell
 */

int _grd3d_fnd_near_cell (
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
    char  s[24]="grd3d_fnd_near_cell";
    int   ir, ix, jy, kz, ii, jj, kk, ia, ib, ok, maxrad;
    int   ixmin, ixmax, jymin, jymax, kzmin, kzmax;
    float corners[24];

    xtgverbose(debug);

    xtg_speak(s,1,"==== Entering routine %s ====",s);

    /* 
     * The quest for the search is to go from I,J,K
     * and seek out as a "square blowing balloon"
     */
    
    maxrad=nx;
    if (ny>maxrad) maxrad=ny;
    if (nz>maxrad) maxrad=nz;
    
    xtg_speak(s,1,"Maxrad is %d", maxrad);

    ix = *i; jy = *j; kz = *k;

    xtg_speak(s,1,"I J K are %d %d %d", ix, jy, kz);
    xtg_speak(s,1,"NX NY NZ %d %d %d", nx, ny, nz);

    /* 
     * ------------------------------------------------------------------------
     * check the current cell
     * ------------------------------------------------------------------------
     */


    /* get the cell corners */
    grd3d_corners(ix,jy,kz,nx,ny,nz,
		  p_coord_v, p_zgrd3d_v,
		  corners, debug);
    
    
    /* check if point is inside current cell */
    
    ib=x_ijk2ib(ix,jy,kz,nx,ny,nz,0);
    ia=p_actnum_v[ib];	    
    ok=x_chk_point_in_cell(x,y,z,corners,1,debug);
    
    if (ok>0) {
	*i=ix; *j=jy; *k=kz;
	xtg_speak(s,1,"Value found in current cell");
	return ia; /* 1 if active, 0 if inactive */
    }
			

    /* 
     * ------------------------------------------------------------------------
     * ...or, check the other cells in a close neighbourhood. If they
     * are not there, then they ARE not there ... (assuming that map
     * resolution is much smaller than 3D grid XY resolution : Hnece it must
     * be returned a code that nothing was found
     * ------------------------------------------------------------------------
     */

    ixmin=ix-1;
    ixmax=ix+1;
    jymin=jy-1;
    jymax=jy+1;
    if (ixmin<1)  ixmin=1;
    if (ixmax>nx) ixmax=nx;
    if (jymin<1)  jymin=1;
    if (jymax>ny) jymax=ny;
    kzmin=1;     /* z needs flexibility due to faults */
    kzmax=nz;


    for (ii=ixmin; ii<=ixmax; ii++) {
	for (jj=jymin; jj<=jymax; jj++) {
	    for (kk=kzmin; kk<=kzmax; kk++) {

		/* get the cell corners */
		grd3d_corners(ii,jj,kk,nx,ny,nz,
			      p_coord_v, p_zgrd3d_v,
			      corners, debug);
		
		
		/* check if point is inside cell */
		
		ib=x_ijk2ib(ii,jj,kk,nx,ny,nz,0);
		ia=p_actnum_v[ib];	    
		ok=x_chk_point_in_cell(x,y,z,corners,1,debug);
		
		if (ok>0) {
		    *i=ii; *j=jj; *k=kk;
		    return ia; /* 1 if active, 0 if inactive */
		    
		}

	    }
	}
    }
    

    
    xtg_speak(s,4,"==== Exiting routine ====");
    
    /* nothing found */
    return -1;
}


