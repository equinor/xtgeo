/*
 *******************************************************************************
 *
 * NAME:
 *    map_reduce_byminmax.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Reduce map by min/max window. There is a scan mode and an a execute mode
 *
 * ARGUMENTS:
 *    mode                 i     0 = scan mode; 1 = execute
 *    nx, ny               i     Map dimension
 *    xori ... yinc        i     Map geometry
 *    p_zval_v             i     Input map (pointer)
 *    newxmin...newxmax    i     New x and y window
 *    nnx, nny            i/o    New map dimensions
 *    nnxsh, nnysh        i/o    Shift in start nodes (to be used in mode=1)
 *    nxori, nyori        i/o    New map geometry
 *    p_newzval_v          o     New map
 *    debug                i     Debug level
 *
 * RETURNS:
 *    Void + new dimensions, geometry and pointer to new map
 *
 * TODO/ISSUES/BUGS:
 *
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 *******************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"


void map_reduce_byminmax (
			  int mode,
			  int nx,
			  int ny,
			  double xori,
			  double yori,
			  double xinc,
			  double yinc,
			  double *p_zval_v,
			  double newxmin,
			  double newxmax,
			  double newymin,
			  double newymax,
			  int *nnx,
			  int *nny,
			  int *nnxsh,
			  int *nnysh,
			  double *nxori,
			  double *nyori,
			  double *p_newzval_v,
			  int debug
			  )
{
    int    i, j;
    char   s[24]="map_reduce_byminmax";
    double newxori, newyori, xpos, ypos;
    int    nxshift, nyshift, nxmax, nymax;
    int    mx, my, ib, ibx, mxs, mys, ii, jj;


    xtgverbose(debug);

    xtg_speak(s,2,"Reduce map...");


    /*
     * SCAN mode, find dimensions
     */
    if (mode == 0) {

	nxshift = nx;
	nyshift = ny;

	nxmax = 1;
	nymax = 1;

	newxori = VERYLARGEFLOAT;
	newyori = VERYLARGEFLOAT;

	for (j=1;j<=ny;j++) {
	    for (i=1;i<=nx;i++) {

		xpos=xori+xinc*(i-1);
		ypos=yori+yinc*(j-1);

		if (xpos >= newxmin && xpos <= newxmax) {
		    if (xpos <= newxori) newxori = xpos;
		    if (i<nxshift) nxshift=i;
		    if (i>nxmax) nxmax=i;
		    if (debug>3) xtg_speak(s,4,"SHIFT %3d",nxshift);
		}

		if (ypos >= newymin && ypos <= newymax) {
		    if (ypos <= newyori) newyori = ypos;
		    if (j<nyshift) nyshift=j;
		    if (j>nymax) nymax=j;
		}

	    }
	}

	if (nxmax<nxshift || nymax<nyshift) {
	    xtg_error(s,"Error 3421. Contact JRIV");
	}

	/* compute new nx and ny etc */
	*nnx = nxmax-nxshift+1;
	*nny = nymax-nyshift+1;

	*nnxsh = nxshift;
	*nnysh = nyshift;

	*nxori = newxori;
	*nyori = newyori;
    }

    /*
     * EXECUTE mode, do the map reduce
     */

    else{
	mx  = *nnx;     /* dimen of new maps */
	my  = *nny;
	mxs = *nnxsh;   /* shift relative to oold map */
	mys = *nnysh;

	xtg_speak(s,2,"New map dimens are %d %d", mx, my);
	xtg_speak(s,2,"Shifts are %d %d", mxs, mys);


	for (j=1; j<=ny; j++) {
	    for (i=1; i<=nx; i++) {

		ib  = x_ijk2ib(i,j,1,nx,ny,1,0);

		if (i >= mxs && i < (mxs+mx) &&
		    j >= mys && j < (mys+my)) {

		    ii=i-mxs+1;
		    jj=j-mys+1;

		    xtg_speak(s,3,"Node %d %d is mapped to %d %d",i,j,ii,jj);

		    ibx = x_ijk2ib(ii,jj,1,mx,my,1,0);

		    p_newzval_v[ibx] = p_zval_v[ib];
		}

	    }
	}
    }
}
