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
 *    cf. XTGeo LICENSE
 ******************************************************************************
 */
#include <math.h>
#include <stdio.h>

#include "libxtg.h"
#include "libxtg_.h"


/*
 *****************************************************************************
 * Private function
 *****************************************************************************
 */
int _interp_holes(int mx, int my, double *p_zval_v, int option,
                  int debug)
{

    int i, ii, j, jj, ib=0, ibn=0;
    double px1, px2, py1, py2, px1w, px2w, py1w, py2w, sumw;
    char s[24]="map_interp_holes";

    xtgverbose(debug);

    /* test every point */

    for (j=1;j<=my;j++) {
	for (i=1;i<=mx;i++) {

	    ib=x_ijk2ib(i,j,1,mx,my,1,0);

	    /* find UNDEF values*/
	    if (p_zval_v[ib] > UNDEF_MAP_LIMIT) {
		xtg_speak(s,3,"Hole for node %d %d found ...",i,j);

		px1=0.0; px1w=VERYLARGEFLOAT;
		px2=0.0; px2w=VERYLARGEFLOAT;
		py1=0.0; py1w=VERYLARGEFLOAT;
		py2=0.0; py2w=VERYLARGEFLOAT;

		for (ii=i; ii>=1; ii--) {
		    ibn=x_ijk2ib(ii,j,1,mx,my,1,0);
		    if (p_zval_v[ibn]<UNDEF_MAP_LIMIT) {
			px1=p_zval_v[ibn];
			px1w=i-ii;
			ii=0; /* to quit the loop */
		    }
		}

		for (ii=i; ii<=mx; ii++) {
		    ibn=x_ijk2ib(ii,j,1,mx,my,1,0);
		    if (p_zval_v[ibn]<UNDEF_MAP_LIMIT) {
			px2=p_zval_v[ibn];
			px2w=ii-i;
			ii=mx+1;
		    }
		}

		for (jj=j; jj>=1; jj--) {
		    ibn=x_ijk2ib(i,jj,1,mx,my,1,0);
		    if (p_zval_v[ibn]<UNDEF_MAP_LIMIT) {
			py1=p_zval_v[ibn];
			py1w=j-jj;
			jj=0;
		    }
		}



		for (jj=j; jj<=my; jj++) {
		    ibn=x_ijk2ib(i,jj,1,mx,my,1,0);
		    if (p_zval_v[ibn]<UNDEF_MAP_LIMIT) {
			py2=p_zval_v[ibn];
			py2w=jj-j;
			jj=my+1;
		    }
		}



		/* now I have potentially 4 values, with 4 weights
		   indicating distance in each direction
		   e.g.
		   px1 = 2400 px1w=3 weight=(1/3)
		   px2 = 3000 px2w=8 --> 1/8
		   py1 = 2600 py1w=1 --> 1/1
		   py2 = 2800 py2w=5 --> 1/5

		   px1 shall have 1/3 influence, px2 1/8, py1 1/1 py2 1/5

		   sum of scales are 1/3+1/8+1/1+1/5 = X = 1.65833

		   hence p1xw_actual = 0.333/1.65833=0.20098
		   hence p2xw_actual = 0.125/1.65833=0.07538
		   hence p1xw_actual = 1/1.65833=0.6030
		   hence p1xw_actual = 0.2/1.65833=0.1206
		   0.20098 + 0.07538+0.6030+0.1206=1.0 ...?

		*/

		px1w=1.0/px1w; px2w=1.0/px2w; py1w=1.0/py1w; py2w=1.0/py2w;
		sumw=px1w+px2w+py1w+py2w;
		px1w=px1w/sumw; px2w=px2w/sumw; py1w=py1w/sumw; py2w=py2w/sumw;

		sumw=px1w+px2w+py1w+py2w;
		if (sumw<0.98 || sumw> 1.02) {
		    xtg_error(s,"Wrong sum for weights. STOP");
		}

		/* assign value */
		p_zval_v[ib]=px1*px1w+px2*px2w+py1*py1w+py2*py2w;
	    }
	}
    }

    return 0;
}



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
    _interp_holes(mx, my, map_v,0,debug);
    xtg_speak(s,1,"Interpolating nodes...DONE");



    xtg_speak(s,2,"Exit from routine <%s> ...",s);

}
