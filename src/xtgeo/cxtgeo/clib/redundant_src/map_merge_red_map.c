/*
 *******************************************************************************
 *
 * NAME:
 *    map_merge_red_map.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Marge a subset map (reduced) into the original
 *
 * ARGUMENTS:
 *    nx1, ny1               i     Map dimension orig map
 *    xori1 ... yinc1        i     Map geometry orig map
 *    p_zval1_v             i/o    Input map (pointer) orig map
 *    nx2, ny2               i     Map dimension other map
 *    xori2 ... yinc2        i     Map geometry other map
 *    p_zval2_v              i     Input map (pointer) other map
 *    debug                  i     Debug level
 *
 * RETURNS:
 *    Void + updated pointer to original map
 *
 * TODO/ISSUES/BUGS:
 *
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 *******************************************************************************
 */

#include <math.h>
#include "libxtg.h"
#include "libxtg_.h"


void map_merge_red_map (
			int nx1,
			int ny1,
			double xori1,
			double yori1,
			double xinc1,
			double yinc1,
			double *p_zval1_v,
			int nx2,
			int ny2,
			double xori2,
			double yori2,
			double xinc2,
			double yinc2,
			double *p_zval2_v,
			int debug
			)
{
    int    i, j, ii, jj, is, js, ib, ibx;
    double xpos1, ypos1, xpos2, ypos2;

    char   s[24]="map_merge_red_map";


    xtgverbose(debug);

    xtg_speak(s,1,"Merge in map...");


    /*
     * First, find where the reduced map starts
     */

    is=0;
    js=0;


    for (j=1;j<=ny1;j++) {
	ypos1=yori1+yinc1*(j-1);

	for (jj=1;jj<ny2;jj++) {
	    ypos2=yori2+yinc2*(jj-1);

	    if (fabs(ypos1-ypos2)<0.00001) {
		js=j;
		goto xloop;
	    }
	}
    }

 xloop:

    for (i=1;i<=nx1;i++) {
	xpos1=xori1+xinc1*(i-1);

	for (ii=1;ii<nx2;ii++) {
	    xpos2=xori2+xinc2*(ii-1);

	    if (fabs(xpos1-xpos2)<0.00001) {
		is=i;
		goto execute;
	    }
	}
    }

 execute:

    /*
     * Do the work
     */

    //xtg_speak(s,1,"IS is %d and

    for (j=1; j<=ny1; j++) {
	for (i=1; i<=nx1; i++) {

	    ib  = x_ijk2ib(i,j,1,nx1,ny1,1,0);

	    if (i >= is && i < (is+nx2) &&
		j >= js && j < (js+ny2)) {

		ii=i-is+1;
		jj=j-js+1;


		ibx = x_ijk2ib(ii,jj,1,nx2,ny2,1,0);

		p_zval1_v[ib] = p_zval2_v[ibx];
	    }

	}
    }
}
