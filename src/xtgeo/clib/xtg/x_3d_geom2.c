/*
 ******************************************************************************
 *
 * A collection of 3D geometrical vectors, planes, etc (volume 2)
 *
 ******************************************************************************
 */

#include <math.h>
#include "libxtg.h"
#include "libxtg_.h"

/*
 ******************************************************************************
 *
 * NAME:
 *    x_point_line_dist.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Distance between a point (or segment) and a line (3D)
 *    Based on: http://paulbourke.net/geometry/pointlineplane/
 *
 * ARGUMENTS:
 *    x1, y1, z1     i     Defining point 1 on the line
 *    x2, y2, z2     i     Defining point 2 on the line
 *    x3, y3, z3     i     Defining point 3 outside the line
 *    distance       0     The distance returned
 *    option1        i     Options flag1:
 *                             0 = infinite line,
 *                             1 = segment (dsitance will "bend" around ends)
 *                             2 = segment and return error -1 if outside
 *    option2        i     Options flag2:
                               0 = distance positive always,
 *                             1 = positive if right side, negative point left
 *    debug          i     Debug level
 *
 * RETURNS:
 *    Function: 0: upon success. If problems:
 *              1: the two points 1 and 2 are the same (forms no line)
 *              2: the input points forms a line
 *              3: the 1 --> 2 vector is too short
 *             -1: The point is outside the segment (given option1 = 2)
 *    Result distance is updated
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ******************************************************************************
 */
int x_point_line_dist(double x1, double y1, double z1,
		      double x2, double y2, double z2,
		      double x3, double y3, double z3,
		      double *distance,
		      int option1, int option2, int debug)
{
    char    s[24]="x_point_line_dist";
    double  sign, u, dlen, x0, y0, z0;

    xtgverbose(debug);

    xtg_speak(s,3,"Entering %s",s);

    /* some checks */
    if (x1==x2 && y1==y2 && z1==z2) {
	/* the two points forms no line*/
	return(1);
    }

    /* length of segment */
    dlen = sqrt (pow(x2-x1,2) + pow(y2-y1,2) + pow(z2-z1,2));


    if (dlen < 1e-20) {
    	return(3);
    }

    /* find u */
    u = (((x3-x1)*(x2-x1)) + ((y3-y1) * (y2-y1)) + ((z3-z1) * (z2-z1)))/
	pow(dlen,2);

    if (option1 == 2 && (u<0 || u > 1)) {
        return(-1);
    }


    if (option1 == 1) {
	if (u<0) u=0;
	if (u>1) u=1;
    }


    /* this gives the point on the line (or segment): */
    x0 = x1 + u*(x2-x1);
    y0 = y1 + u*(y2-y1);
    z0 = z1 + u*(z2-z1);


    /* the actual distance: */
    dlen = sqrt (pow(x3-x0,2) + pow(y3-y0,2) + pow(z3-z0,2));

    /* give sign according to side seen in XY plane ... */

    sign = 0;
    if (option2 == 1) {
	if (x2>x1) {
	    if (y3>=y0) sign=1;
	    if (y3<y0) sign=-1;
	}
	else if (x2<x1) {
	    if (y3>=y0) sign=-1;
	    if (y3<y0) sign=1;
	}
	else{
	    if (x3>=x0) sign=1;
	    if (x3<x0) sign=-1;
	}
	dlen = dlen*sign;
    }

    *distance = dlen;

    return(0);
}



/*
 ******************************************************************************
 *
 * NAME:
 *    x_point_line_pos.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Projected XYZ on line between a point and a line (3D)
 *    Based on: http://paulbourke.net/geometry/pointlineplane/
 *
 * ARGUMENTS:
 *    x1, y1, z1     i     Defining point 1 on the line
 *    x2, y2, z2     i     Defining point 2 on the line
 *    x3, y3, z3     i     Defining point 3 outside the line
 *    x, y, z        o     The projected point coordinate
 *    rel            o     Relative position (from p1) if segment
 *    option1        i     Options flag1:
 *                             0 = infinite line,
 *                             1 = segment
 *                             2 = segment, allow for numerical precision
 *    debug          i     Debug level
 *
 * RETURNS:
 *    Function: 0: upon success. If problems:
 *              1: the two points 1 and 2 are the same (forms no line)
 *              2: the input points forms a line
 *              3: the 1 --> 2 vector is too short
 *             -1: The point is outside the segment (given option1 = 1 or 2)
 *    Result pointers are updated.
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ******************************************************************************
 */
int x_point_line_pos(double x1, double y1, double z1,
                     double x2, double y2, double z2,
                     double x3, double y3, double z3,
                     double *x, double *y, double *z,
                     double *rel,
                     int option1, int debug)
{
    char    s[24]="x_point_line_pos";
    double  u, dlen, rellen, fullen, x0, y0, z0;

    xtgverbose(debug);

    xtg_speak(s,3,"Entering %s",s);

    /* some checks */
    if (x1==x2 && y1==y2 && z1==z2) {
	/* the two points forms no line*/
	return(1);
    }

    /* length of segment */
    dlen = sqrt (pow(x2-x1,2) + pow(y2-y1,2) + pow(z2-z1,2));


    if (dlen < 1e-20) {
    	return(3);
    }

    /* find u */
    u = (((x3 - x1) * (x2 - x1)) + ((y3 - y1) * (y2 - y1)) +
         ((z3 - z1) * (z2 - z1))) / pow(dlen, 2);

    if (option1 == 1 && (u < 0 || u > 1)) {
        return(-1);
    }

    if (option1 == 2) {
        if (u < (0.0 - FLOATEPS) || u > (1 + FLOATEPS)) return(-1);
        if (u < 0.0) u = 0.0 + FLOATEPS;  /* making edge points being inside */
        if (u > 1.0) u = 1.0 - FLOATEPS;  /* making edge points being inside */
    }

    /* this gives the point on the line (or segment): */
    x0 = x1 + u * (x2 - x1);
    y0 = y1 + u * (y2 - y1);
    z0 = z1 + u * (z2 - z1);


    /* the actual relative distance: */
    rellen = sqrt(pow(x1 - x0, 2) + pow(y1 - y0, 2) + pow(z1 - z0, 2));
    fullen = sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2) + pow(z1 - z2, 2));

    *rel = rellen / fullen;
    *x = x0;
    *y = y0;
    *z = z0;

    return(0);
}
