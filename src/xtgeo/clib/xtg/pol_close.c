/*
 * ############################################################################
 * pol_close.c
 *
 * Description:
 * Close a polygon by adding a point that has sam XYZ as first point
 * Allocation checks are to be done in the caller...
 *
 * Bugs or potential problems: Allocation trouble, checks needed?
 *
 * Author: J.C. Rivenaes JRIV@statoil.com
 * ############################################################################
 */

#include <math.h>
#include "libxtg.h"
#include "libxtg_.h"

int pol_close (
	       int    np,
	       double *p_x_v,
	       double *p_y_v,
	       double *p_z_v,
	       double dist,
	       int    option,
	       int    debug
	       )
{

    double  dist2d, dist3d, usedist;

    xtgverbose(debug);

    char s[24]="pol_close";
    xtg_speak(s,2,"Entering routine...");

    dist2d=sqrt(pow(p_x_v[0]-p_x_v[np-1],2) + pow(p_y_v[0]-p_y_v[np-1],2));
    dist3d=sqrt(pow(p_x_v[0]-p_x_v[np-1],2) + pow(p_y_v[0]-p_y_v[np-1],2) + pow(p_z_v[0]-p_z_v[np-1],2));

    if (option==1) {
	usedist=dist2d;
    }
    else{
	usedist=dist3d;
    }

    if (usedist>0.0 && usedist<dist) {
	p_x_v[np]=p_x_v[0];
	p_y_v[np]=p_y_v[0];
	p_z_v[np]=p_z_v[0];

	xtg_speak(s,2,"Actual distance is %9.3f, maximum distance for closing is %9.3f. OK.", usedist, dist);

	return(np+1);
    }
    else{
	xtg_speak(s,1,"Actual distance is %9.3f, maximum distance is %9.3f is exceeded", usedist, dist);
	return(np);
    }
}
