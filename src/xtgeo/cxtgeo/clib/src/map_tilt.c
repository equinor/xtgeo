/*
 * ############################################################################
 * map_tilt.c
 *
 * Description:
 * Tilt a map by an angle and an azimuth. The tilting is done by leaving
 * the first node in lower left corner untouched. A positive tilt should
 * make it dip down, compared with the first node. An azimuth of zero makes
 * a north-south tilt
 *
 * Bugs or potential problems:
 * -
 *
 * Author: J.C. Rivenaes
 * ############################################################################
 * $Id: map_export_storm_binary.c,v 1.1 2001/03/14 08:02:29 bg54276 Exp $
 * $Source: /d/proj/bg/avresmod/src/gplib/GPLExt/RCS/map_export_storm_binary.c,v $
 *
 * $Log: $
 *
 * ############################################################################
 */

#include <math.h>
#include "libxtg.h"
#include "libxtg_.h"

void map_tilt (
	       int   nx,
	       int   ny,
	       double dx,
	       double dy,
	       double xstart,
	       double ystart,
	       double *p_map_v,
	       double angle,
	       double azimuth,
	       int   ierror,
	       int   debug
	       )
{

    int i, j, ib;
    double a, x, y, rangle, razi, len, dz;
    char s[24];

    strcpy(s,"map_tilt");

    xtgverbose(debug);
    xtg_speak(s,2,"Entering <map_tilt>...");

    rangle = PI*angle/180;   /* radians */
    razi   = PI*azimuth/180; /* radians */
    if (rangle >= (PIHALF-0.01)) {
	xtg_error(s,"Illegal large angle given!");
    }
    xtg_speak(s,2,"Angle is degrees (radians): %7.3f (%7.3f)", angle, rangle);


    for (j=1;j<=ny;j++) {
	for (i=1;i<=nx;i++) {

	    if (i==1 && j==1) {
		dz=0.0;
	    }
	    else{
		/* find the angle (azimuth) of the point compared with xy_first */
		x=dx*(i-1);
		y=dy*(j-1);
		if (y>0.01) {
		    a=atan(x/y);
		}
		else{
		    a=PIHALF;
		}

		/* adjust the angle by azimuth */
		a=a-razi;


		/* find the dz */
		len=sqrt(pow(x,2) + pow(y,2));
		dz=len*tan(rangle)*cos(a);
	    }

	    ib=x_ijk2ib(i,j,1,nx,ny,1,0);

	    if (p_map_v[ib] < UNDEF_MAP_LIMIT) {
		p_map_v[ib]=p_map_v[ib] + dz;
	    }
	}
    }
    xtg_speak(s,2,"Exiting <map_operation_map>...");
}
