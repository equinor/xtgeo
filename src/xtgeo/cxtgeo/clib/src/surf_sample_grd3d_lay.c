/*
 ******************************************************************************
 *
 * NAME:
 *    surf_sample_grd3d_lay.c (based on map_sample_grd3d_lay)
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Sample values from 3D grid layer to regular map (with possible rotation)
 *
 * ARGUMENTS:
 *    nx,ny,nz       i     3D drid dimensions I J K
 *    p_zcorn_v      i     ZCorn values
 *    p_coord_v      i     Coord values
 *    p_actnum_v     i     ACTNUM values
 *    klayer         i     Actual K layer to sample from...
 *    mx, my         i     Map dimension
 *    xori,xinc      i     Maps X settings
 *    yori,yinc      i     Maps Y settings
 *    rotation       i     Map rotation
 *    map_v         i/o    Maps depth values
 *    option         i     0: top cell, 1: base cell
 *    debug          i     Debug level
 *
 * RETURNS:
 *    Void + Changed pointer to map property (z values)
 *
 * TODO/ISSUES/BUGS:
 *    Map rotation is currently NOT supported!
 *
 * LICENCE:
 *    Statoil property
 ******************************************************************************
 */
#include <math.h>
#include <stdio.h>

#include "libxtg.h"
#include "libxtg_.h"

void surf_sample_grd3d_lay (
			   int    nx,
			   int    ny,
			   int    nz,
			   double *p_coord_v,
			   double *p_zcorn_v,
			   int    *p_actnum_v,
			   int    klayer,
			   int    mx,
			   int    my,
			   double xori,
			   double xinc,
			   double yori,
			   double yinc,
                           double rotation,
			   double *map_v,
			   int    option,
			   int    debug
			   )

{
    /* locals */
    char    s[24]="surf_sample_grd3d_lay";
    double  corners_v[24];
    double  zval, xpos, ypos, cxmin, cxmax, cymin, cymax;
    int     mode, ibm, i, j;
    int     mxmin, mxmax, mymin, mymax, ii, jj;
    int     ishift;

    xtgverbose(debug);

    if (rotation != 0.0) {
        xtg_error(s, "Map rotation not supported so far...");
    }

    xtg_speak(s,2,"Entering routine <%s> ...",s);

    xtg_speak(s,3,"NX NY NZ: %d %d %d", nx, ny, nz);
    xtg_speak(s,3,"MX and MY is: %d %d", mx, my);

    /*
     * Loop over 3D cells, finding the min/max edges per cell in XY
     * Then find the map indexing needed to cover that cell and
     * populate these (local) map nodes
     */

    mode = option; /* meaning cell top=0 base=1 */
    ishift=0;
    if (mode==1) ishift=12;

    for (j=1; j<=ny; j++) {
	for (i=1; i<=nx; i++) {

	    /* get the corners for the cell */
	    grd3d_corners(i,j,klayer,nx,ny,nz,p_coord_v,
			  p_zcorn_v, corners_v, debug);

	    /* find cell min/max  both for X and Y */
	    cxmin=999999999;
	    cxmax=-999999999;
	    for (ii=0+ishift; ii<=9+ishift; ii+=3) {
		if (corners_v[ii]<cxmin) cxmin=corners_v[ii];
		if (corners_v[ii]>cxmax) cxmax=corners_v[ii];
	    }

	    cymin=999999999;
	    cymax=-999999999;
	    for (ii=1+ishift; ii<=10+ishift; ii+=3) {
		if (corners_v[ii]<cymin) cymin=corners_v[ii];
		if (corners_v[ii]>cymax) cymax=corners_v[ii];
	    }

	    /* now find the map node range to test for */
	    mxmin = (int)floor(((cxmin-xori)/xinc)+1);
	    mxmax = (int)ceil(((cxmax-xori)/xinc)+1+0.5);
	    mymin = (int)floor(((cymin-yori)/yinc)+1);
	    mymax = (int)ceil(((cymax-yori)/yinc)+1+0.5);


	    if (mxmin<1)  mxmin=1;
	    if (mxmax>mx) mxmax=mx;
	    if (mymin<1)  mymin=1;
	    if (mymax>my) mymax=my;

	    if (j==99950 && i==50) {
		printf("mxmin etc: %d %d %d %d\n", mxmin, mxmax, mymin, mymax);
		printf("cxmin etc: %f %f %f %f\n", cxmin, cxmax, cymin, cymax);
	    }


	    /* now loop over the local map nodes */
	    for (jj = mymin; jj <= mymax; jj++) {
		for (ii = mxmin; ii <= mxmax; ii++) {
		    ibm = x_ijk2ic(ii,jj,1,mx,my,1,0);
		    xpos = xori+xinc*(ii-1);
		    ypos = yori+yinc*(jj-1);

		    zval=x_sample_z_from_xy_cell(corners_v, xpos, ypos,
						 mode, debug);
		    if (zval < UNDEF_LIMIT && zval > -1*UNDEF_LIMIT){
			map_v[ibm]=zval;

		    }
		}
	    }

	}
    }

    /* interpolating undef map nodes */
    xtg_speak(s,1,"Interpolating nodes...");
    map_interp_holes(mx, my, map_v,0,debug);
    xtg_speak(s,1,"Interpolating nodes...DONE");



    xtg_speak(s,2,"Exit from routine <%s> ...",s);

}
