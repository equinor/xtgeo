/*
 * ############################################################################
 * map_chk_point_between.c
 * Checks whether a XY point is beteen 2 maps. The purpose is to eg find
 * that a well point in in correct zone.
 *
 * Its is possible for the maps to have different resolutions/settings
 *
 * Map 1 should lie above map 2 (may be equal, though)
 *
 * Author: J.C. Rivenaes
 * ############################################################################
 * $Id: $
 * $Source: $
 *
 * $Log: $
 *
 * ############################################################################
 * General description:
 * OUTSIDE= 0 you are inside. ZDIFF is now fraction from top z-z1/z2-z1
 * OUTSIDE=-1 you are above.  ZDIFF is measured from above top surf
 * OUTSIDE= 1 you are below.  ZDIFF is measured from below lower surf
 * OUTSIDE=-9 inconsistent maps...
 * ############################################################################
 * TODO:
 * - proper handling of undefined map nodes
 * ############################################################################
 */

#include <math.h>
#include "libxtg.h"
#include "libxtg_.h"

/*
 * ****************************************************************************
 *                         GRD2D_CHK_POINT_BETWEEN
 * Scan the maps (1) and (2) and find the point given by x,y,z. Finds the
 * corresponding z1,z2 in the maps
 *
 * ****************************************************************************
 *
 */
void map_chk_point_between(
			     double x,
			     double y,
			     double z,

			     int nx1,
			     int ny1,
			     double xstep1,
			     double ystep1,
			     double xmin1,
			     double ymin1,
			     double *p_map1_v,

			     int nx2,
			     int ny2,
			     double xstep2,
			     double ystep2,
			     double xmin2,
			     double ymin2,
			     double *p_map2_v,

			     int   *outside,
			     double *zdiff,

			     int debug
			     )
{
    int    i, j, m, n, ib, ic;
    double  xa, ya, xb, yb, zp1=0.0, zp2=0.0, xx;
    double  x_v[4], y_v[4], z_v[4];
    char sub[24]="map_chk_point_between";

    xtgverbose(debug);

    xtg_speak(sub,3,"Entering routine");



    /* map 1 */
    for (j=1;j<ny1;j++) {
	for (i=1;i<nx1;i++) {
	    xa=xmin1 + (i-1)*xstep1;
	    ya=ymin1 + (j-1)*ystep1;
	    xb=xmin1 + (i-0)*xstep1;
	    yb=ymin1 + (j-0)*ystep1;


	    if (x >= xa && x < xb  && y >= ya && y < yb) {
		/* found the map nodes that makes a square around...*/
		/* need the z, y and depths, as arrays */
		ic=0;
		for (n=1;n>=0;n--) {
		    for (m=1;m>=0;m--) {
			ib=x_ijk2ib(i-m+1,j-n+1,1, nx1,ny1,1,0);
			x_v[ic]=xmin1 + (i-m)*xstep1;
			y_v[ic]=ymin1 + (j-n)*ystep1;
			z_v[ic]=p_map1_v[ib];
			ic++;
		    }
		}

		/* find the depths ... */
		zp1=x_interp_map_nodes(x_v, y_v, z_v, x, y, 2, debug);

		if (debug >= 3) {
		    xtg_speak(sub,3,"ZP1 in map 1 is %9.2f",zp1);
		}

	    }
	}
    }


    /* map 2 */
    for (j=1;j<ny2;j++) {
	for (i=1;i<nx2;i++) {
	    xa=xmin2 + (i-1)*xstep2;
	    ya=ymin2 + (j-1)*ystep2;
	    xb=xmin2 + (i-0)*xstep2;
	    yb=ymin2 + (j-0)*ystep2;

	    if (x >= xa && x < xb  && y >= ya && y < yb) {
		/* found the map nodes that makes a square around...*/

		/* need the z, y and depths, as arrays */
		ic=0;
		for (n=1;n>=0;n--) {
		    for (m=1;m>=0;m--) {
			ib=x_ijk2ib(i-m+1,j-n+1,1, nx1,ny1,1,0);
			x_v[ic]=xmin2 + (i-m)*xstep2;
			y_v[ic]=ymin2 + (j-n)*ystep2;
			z_v[ic]=p_map2_v[ib];
			ic++;
		    }
		}

		/* find the depths ... */
		zp2=x_interp_map_nodes(x_v, y_v, z_v, x, y, 2, debug);

		if (debug >= 3) {
		    xtg_speak(sub,3,"ZP2 in map 2 is %9.2f",zp2);
		}
	    }
	}
    }

    /*
     *-------------------------------------------------------------------------
     * Status
     * OUTSIDE= 0 you are inside. ZDIFF is now fraction from top z-z1/z2-z1
     * OUTSIDE=-1 you are above.  ZDIFF is measured from above top surf
     * OUTSIDE= 1 you are below.  ZDIFF is measured from below lower surf
     * OUTSIDE=-9 inconsistent maps...
     *-------------------------------------------------------------------------
     */
    *outside=0;

    xtg_speak(sub,3,"Z, ZP1, ZP2:  %9.2f  %9.2f  %9.2f",z,zp1,zp2);

    if (zp2 < zp1) {
	xtg_warn(sub,1,"Maps are inconsistent! will not compute!");
	*outside=-9;
	return;
    }


    /* above the upper surface ...*/
    if (z < zp1) {
	*outside=-1;
	xx=zp1-z;
	*zdiff=zp1-z;
    }
    /* below the lower surface ...*/
    else if (z > zp2) {
	*outside=1;
	xx=z-zp2;
	*zdiff=z-zp2;
    }
    /* inbetween - give fraction of thickness meas. from upper*/
    else {
	if (fabs(zp2-zp1)>0.01) {
	    *zdiff=(z-zp1)/(zp2-zp1);
	}
	else {
	    *zdiff=0.0;
	}
    }

}
