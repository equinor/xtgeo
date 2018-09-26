/*
 * ############################################################################
 * pol_extend.c
 *
 * Description:
 * Extend a polygon by a given max distance, in one of the ends, or both
 *
 * Arguments:
 * np             Number of points present (0..(np-1))
 * p_x_v...p_z_v  Polygon arrays
 * dist           Distance to extend
 * mode           1: beginning; 2; end extend; 3 both
 * xang           Additional angle (in radians)
 * option         Presently unused
 * debug          Verbose flag
 *
 * Author: J.C. Rivenaes JRIV@statoil.com
 * ############################################################################
 */

#include "libxtg.h"
#include "libxtg_.h"

int pol_extend (
		int    np,
		double *p_x_v,
		double *p_y_v,
		double *p_z_v,
		double dist,
		int    mode,
		double xang,
		int    option,  /* 0: look in 3D, 1: look in 2d XY */
		int    debug
		)
{

    double  x, y, z;
    int i;
    int    iostat;

    xtgverbose(debug);

    char s[24]="pol_extend";
    xtg_speak(s,2,"Entering routine %s with mode %d", s, mode);



    if (mode==1 || mode==3) {

	/* move the previous point to the next to make a vacant space at location 0 */
	for (i=np;i>0;i--) {
	    p_x_v[i]=p_x_v[i-1];
	    p_y_v[i]=p_y_v[i-1];
	    p_z_v[i]=p_z_v[i-1];
	}

	x=p_x_v[1];
	y=p_y_v[1];
	z=p_z_v[1];

	xtg_speak(s,2,"MODE1: Point XY is %10.2f %10.2f", x, y);

	/* this will update the first point by moving it a dist length */
	iostat=x_vector_extrapol(p_x_v[2],p_y_v[2],p_z_v[2], &x, &y, &z,
				 dist, xang, debug);

        if (iostat != 1) xtg_error(s, "Error from %s", s);

        p_x_v[0] = x;
	p_y_v[0] = y;
	p_z_v[0] = z;

	xtg_speak(s,2,"MODE1: Updated Point XY is %10.2f %10.2f", x, y);

	np+=1;

    }
    if (mode==2 || mode==3) {

	x=p_x_v[np-1];
	y=p_y_v[np-1];
	z=p_z_v[np-1];


	/* this will update the last point by moving it a dist length */
	x_vector_extrapol(p_x_v[np-2],p_y_v[np-2],p_z_v[np-2], &x, &y, &z,
			  dist, xang, debug);

	p_x_v[np] = x;
	p_y_v[np] = y;
	p_z_v[np] = z;

	np+=1;

    }

    return(1);

}
