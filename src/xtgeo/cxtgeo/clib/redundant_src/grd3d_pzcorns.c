/*
 * ############################################################################
 * Calculating a pillar corner Z coordiate. This is the cross point at a pillar
 * node (in contrast to cell corners of one cell, cf grd3d_corners.c
 * The i,j,k numbering refers to the range 1 to nx+1 for each row; similar for 
 * columns and layers.
 * It returns a flag + an array with 4 numbers. Array index 0 is corner 1.
 * In case of unfaulted, numbers are equal. 
 * In case of faulted, these are different. A flag == 1 will also mark 
 * faulted pcorner.
 *
 *    2 | 4
 *  ---------  
 *    1 | 3        If corners are at edges they
 *                 will get an UNDEF value
 *                                                         i,j+1    i+1,j+1 
 * To be clear, a cell centre i,j is surrounder by corners   i,j    i+1,j
 *                                                         
 * Author: JCR
 * ############################################################################
 */

#include <string.h>
#include <stdlib.h>
#include "libxtg.h"
#include "libxtg_.h"


void grd3d_pzcorns (
		    int      i,
		    int      j,
		    int      k,
		    int      nx,
		    int      ny,
		    int      nz,
		    double   *p_coord_v,
		    double   *p_zgrd3d_v,
		    double   *p,
		    int      *flag,
		    int      debug
		    )


{
    int ic, ib;
    char  s[24]="grd3d_pcorners";

    xtgverbose(debug);

    xtg_speak(s,4,"==== Entering grd3d_pzcorns ====");

    *flag=0; /* assume nonfaulted */

    for (ic=0;ic<=3;ic++) {
	p[ic]=UNDEF;
    }


    if (i==1 && j==1) {
	/* pzcorn 4 of corner 1 in cell i,j */
	ib=x_ijk2ib(i,j,k,nx,ny,nz+1,0);
	p[3]= p_zgrd3d_v[4*ib + 1*1 - 1];
    }
    else if (i==1 && j>1 && j<ny+1) {
	/* pzcorn 3 of corner 2 in cell i,j-1 */
	ib=x_ijk2ib(i,j-1,k,nx,ny,nz+1,0);
	p[2]= p_zgrd3d_v[4*ib + 1*2 - 1];
	/* pzcorn 4 of corner 1 in cell i,j */
	ib=x_ijk2ib(i,j,k,nx,ny,nz+1,0);
	p[3]= p_zgrd3d_v[4*ib + 1*1 - 1];
    }
    else if (i==nx+1 && j>1 && j<ny+1) {
	/* pzcorn 1 of corner 4 in cell i-1,j-1 */
	ib=x_ijk2ib(i-1,j-1,k,nx,ny,nz+1,0);
	p[0]= p_zgrd3d_v[4*ib + 1*4 - 1];
	/* pzcorn 2 of corner 3 in cell i-1,j */
	ib=x_ijk2ib(i-1,j,k,nx,ny,nz+1,0);
	p[1]= p_zgrd3d_v[4*ib + 1*3 - 1];
    }
    else if (i==1 && j==ny+1) {
	/* pzcorn 3 of corner 2 only of cell i,j-1 */
	ib=x_ijk2ib(i,j-1,k,nx,ny,nz+1,0);
	p[2]= p_zgrd3d_v[4*ib + 1*2 - 1];
    }
    else if (i>1 && i<nx+1 && j==1) {
	/* pzcorn 2 of corner 3 cell i-1,j */
	ib=x_ijk2ib(i-1,j,k,nx,ny,nz+1,0);
	p[1]= p_zgrd3d_v[4*ib + 1*3 - 1];
	/* pzcorn 4 of corner 1 cell i,j */
	ib=x_ijk2ib(i,j,k,nx,ny,nz+1,0);
	p[3]= p_zgrd3d_v[4*ib + 1*1 - 1];
    }
    else if (i>1 && i<nx+1 && j==ny+1) {
	/* pzcorn 1 of corner 4 of cell i-1,j-1 */
	ib=x_ijk2ib(i-1,j-1,k,nx,ny,nz+1,0);
	p[0]= p_zgrd3d_v[4*ib + 1*4 - 1];
	/* pzcorn 3 of corner 2 of cell i,j-1 */
	ib=x_ijk2ib(i,j-1,k,nx,ny,nz+1,0);
	p[2]= p_zgrd3d_v[4*ib + 1*2 - 1];
    }
    else if (i==nx+1 && j==1) {
	/* pzcorn 2 is corner 3 of cell i-1,j */
	ib=x_ijk2ib(i-1,j,k,nx,ny,nz+1,0);
	p[1]= p_zgrd3d_v[4*ib + 1*3 - 1];
    }
    else if (i==nx+1 && j==ny+1) {
	/* pzcorn 1 is corner 4 of cell i-1,j-1 */
	ib=x_ijk2ib(i-1,j-1,k,nx,ny,nz+1,0);
	p[0]= p_zgrd3d_v[4*ib + 1*4 - 1];
    }
    else{
	/* pzcorn 1(count=0) is corner 4 of cell i-1,j-1 */
	ib=x_ijk2ib(i-1,j-1,k,nx,ny,nz+1,0);
	p[0]= p_zgrd3d_v[4*ib + 1*4 - 1];
	/* pzcorn 2 is corner 3 of cell i-1,j */
	ib=x_ijk2ib(i-1,j,k,nx,ny,nz+1,0);
	p[1]= p_zgrd3d_v[4*ib + 1*3 - 1];
	/* pzcorn 3 is corner 2 of cell i,j-1 */
	ib=x_ijk2ib(i,j-1,k,nx,ny,nz+1,0);
	p[2]= p_zgrd3d_v[4*ib + 1*2 - 1];
	/* pzcorn 4 is corner 1 of cell i,j */
	ib=x_ijk2ib(i,j,k,nx,ny,nz+1,0);
	p[3]= p_zgrd3d_v[4*ib + 1*1 - 1];

    }

    /* flag unfinished */

    xtg_speak(s,4,"==== Exiting grd3d_pzorns ====");

}


