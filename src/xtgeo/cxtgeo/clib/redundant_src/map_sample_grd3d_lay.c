/*
 ******************************************************************************
 *
 * NAME:
 *    map_sample_grd3d_lay.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Sample values from 3D grid layer to a regular map
 *
 * ARGUMENTS:
 *    nx,ny,nz       i     3D drid dimensions I J K
 *    p_zcorn_v      i     ZCorn values
 *    p_coord_v      i     Coord values
 *    p_actnum_v     i     ACTNUM values
 *    klayer         i     Actual K layer to sample from...
 *    mx, my         i     Map dimension
 *    xmin,xstep     i     Maps X settings
 *    ymin,ystep     i     Maps Y settings
 *    p_zval_v      i/o    Maps depth values
 *    option         i     Options flag for later usage
 *    debug          i     Debug level
 *
 * RETURNS:
 *    Void + Changed pointer to map property (z values)
 *
 * TODO/ISSUES/BUGS:
 *
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ******************************************************************************
 */


#include <math.h>
#include "libxtg.h"
#include "libxtg_.h"

/*
 * ############################################################################
 * Input a depth map, returns an map array that holds the ib's for the 3D grid:
 * p_ib_v[mapnode] = ib_cell_number; otherwise undefined
 * ############################################################################
 */

void map_sample_grd3d_lay (
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
			   double xstep,
			   double yori,
			   double ystep,
			   double *p_zval_v,
			   int    option,
			   int    debug
			   )

{
    /* locals */
    char    s[24]="map_sample_grd3d_lay";
    double  corners_v[24];
    double  zval, xpos, ypos, cxmin, cxmax, cymin, cymax;
    int     mode, ibm, i, j;
    int     mxmin, mxmax, mymin, mymax, ii, jj;
    int     ishift;


    xtgverbose(debug);

    xtg_speak(s,2,"Entering routine <%s> ...",s);

    xtg_speak(s,3,"NX NY NZ: %d %d %d", nx, ny, nz);
    xtg_speak(s,3,"MX and MY is: %d %d", mx, my);

    /* reset map to all undef nodes */
    map_operation_value(-1,mx,my,p_zval_v,UNDEF,UNDEF,UNDEF,debug);


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
	    mxmin = (int)floor(((cxmin-xori)/xstep)+1);
	    mxmax = (int)ceil(((cxmax-xori)/xstep)+1+0.5);
	    mymin = (int)floor(((cymin-yori)/ystep)+1);
	    mymax = (int)ceil(((cymax-yori)/ystep)+1+0.5);


	    if (mxmin<1)  mxmin=1;
	    if (mxmax>mx) mxmax=mx;
	    if (mymin<1)  mymin=1;
	    if (mymax>my) mymax=my;

	    if (j==99950 && i==50) {
		printf("mxmin etc: %d %d %d %d\n", mxmin, mxmax, mymin, mymax);
		printf("cxmin etc: %f %f %f %f\n", cxmin, cxmax, cymin, cymax);
	    }


	    /* now loop over the local map nodes */
	    for (jj=mymin;jj<=mymax;jj++) {
		for (ii=mxmin;ii<=mxmax;ii++) {
		    ibm=x_ijk2ib(ii,jj,1,mx,my,1,0);
		    xpos=xori+xstep*(ii-1);
		    ypos=yori+ystep*(jj-1);

		    if (j==99950 && i==50) {
			printf("XPOS YPOS %f %f\n", xpos, ypos);
		    }

		    zval=x_sample_z_from_xy_cell(corners_v, xpos, ypos,
						 mode, debug);
		    if (zval < UNDEF_LIMIT && zval > -1*UNDEF_LIMIT){
			p_zval_v[ibm]=zval;

			if (j==99950 && i==50) {
			    printf("ZVAL %f\n", zval);
			}

		    }
		}
	    }

	}
    }

    /* interpolating undef map nodes */
    xtg_speak(s,1,"Interpolating nodes...");
    map_interp_holes(mx, my, p_zval_v,0,debug);
    xtg_speak(s,1,"Interpolating nodes...DONE");



    xtg_speak(s,2,"Exit from routine <%s> ...",s);

}
