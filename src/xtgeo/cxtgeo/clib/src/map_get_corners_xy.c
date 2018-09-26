/*
 * ############################################################################
 * map_get_corners_xy.c
 * Finds the IB of an I,J coordinate of the lower left corner in a map, given
 * an XY value
 * 
 *   |-------|
 *   | *     |
 * ->|_______|
 *   ^
 *   |
 *
 * Hence the point is surrounded by i,i+1,j,j+1
 *
 * Author: J.C. Rivenaes
 * ############################################################################
 * $Id: $ 
 * $Source: $ 
 *
 * $Log: $
 *
 * ############################################################################
 * TODO:
 * - proper handling of undefined map nodes
 * ############################################################################
 *
 */

#include "libxtg.h"
#include "libxtg_.h"

int map_get_corners_xy(
			 double x,
			 double y,			 
			 int nx,
			 int ny,
			 double xstep, 
			 double ystep,
			 double xmin, 
			 double ymin, 
			 double *p_map_v,
			 
			 int debug
			 )
{
    int    ic, jc, ib=-9;
    double  xa, ya, xb, yb;
    char sub[24]="map_get_corners_xy";

    xtgverbose(debug);

    xtg_speak(sub,4,"Entering routine");

    xtg_speak(sub,4,"x y %f %f ",x,y);
    xtg_speak(sub,4,"xmin ymin xstep ystep nx ny %f %f %f %f %d %d",xmin,ymin,xstep,ystep,nx,ny);

    ib=-1;
    
    /* scan map */
    for (jc=1;jc<ny;jc++) {
	for (ic=1;ic<nx;ic++) {
	    xa=xmin + (ic-1)*xstep;
	    ya=ymin + (jc-1)*ystep;
	    xb=xmin + (ic-0)*xstep;
	    yb=ymin + (jc-0)*ystep;
	    
	    if (x >= xa && x < xb  && y >= ya && y < yb) {
		/* found the map nodes that makes a square around...*/
		ib=x_ijk2ib(ic,jc,1,nx,ny,1,0);
		xtg_speak(sub,4,"Returning value IB: %6d",ib);
		xtg_speak(sub,4,"For ic, jc %6d %6d",ic, jc);
		xtg_speak(sub,4,"X and Y is %11.2f %11.2f",x,y);
		xtg_speak(sub,4,"XA and YA is %11.2f %11.2f",xa,ya);
		xtg_speak(sub,4,"XB and YB is %11.2f %11.2f",xb,yb);
		return ib;
	    }
	}
    }
    xtg_speak(sub,4,"No cell found!");
    return ib; /* returned here if failure; a negative value */
}




