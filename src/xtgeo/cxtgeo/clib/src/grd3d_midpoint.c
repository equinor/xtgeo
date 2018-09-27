/*
 * ############################################################################
 * Calculating XYZ midpoint for a cell
 * TODO: Check just average is OK algorithm?
 * Author: JCR
 * ############################################################################
 */


#include <string.h>
#include <stdlib.h>
#include "libxtg.h"
#include "libxtg_.h"


void grd3d_midpoint (
		     int     i,
		     int     j,
		     int     k,
		     int     nx,
		     int     ny,
		     int     nz,
		     double  *p_coord_v,
		     double  *p_zgrd3d_v,
		     double  *x,
		     double  *y,
		     double  *z,
		     int     debug
		     )


{
    double c[24];
    char   s[24]="grd3d_midpoint";
    
    xtgverbose(debug);
    
    xtg_speak(s,3,"Entering routine %s", s);
    xtg_speak(s,3,"I %d   J %d   K %d   NX %d   NY %d    NZ %d", i,j,k,nx,ny,nz);

    
    /* get all 24 corners */
    xtg_speak(s,3,"Corners...");

    grd3d_corners(i,j,k,nx,ny,nz,p_coord_v,p_zgrd3d_v,c,debug);
    xtg_speak(s,3,"Corners... DONE");

    /* find the midpoint for X,Y,Z (is this OK algorithm?)*/
       
    *x=0.125*(c[0]+c[3]+c[6]+c[9]+c[12]+c[15]+c[18]+c[21]);
    *y=0.125*(c[1]+c[4]+c[7]+c[10]+c[13]+c[16]+c[19]+c[22]);
    *z=0.125*(c[2]+c[5]+c[8]+c[11]+c[14]+c[17]+c[20]+c[23]);

    xtg_speak(s,4,"Midpoint is: %f %f %f",*x, *y, *z);

    xtg_speak(s,4,"==== Exiting ====");

}


