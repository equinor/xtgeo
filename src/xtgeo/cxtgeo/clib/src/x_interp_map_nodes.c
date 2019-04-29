/*
 *******************************************************************************
 *
 * NAME:
 *    x_interp_map_node.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    This routine finds the interpolation within a set of 4 map nodes
 *    The routine assumes that the point is inside the nodes given by x_v,
 *    y_v, and z_v. This must be checked by the calling routine, but a
 *    test here is also provided.
 *
 * The points should be organized as follows (nonrotated maps):
 *
 *     2       3          N
 *                        |
 *     0       1          |___E
 *
 * ARGUMENTS:
 *    x_v, y_v, z_v  i     Coordiantes for XYZ, 4 corners
 *    x, y           i     Defining point inside map cell
 *    method         i     1: my own (not that good?)
 *                         2: bilinear interpolation nonrotated mesh
 *                         3: bilinear interpolation general/rotated mesh
 *    debug          i     Debug level
 *
 * RETURNS:
 *    Z value upon success, but UNDEF_MAP if problems
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 *******************************************************************************
 */

#include <math.h>
#include "libxtg.h"
#include "libxtg_.h"


double x_interp_map_nodes (
			  double *x_v,
			  double *y_v,
			  double *z_v,
			  double x,
			  double y,
			  int method,
			  int debug
			  )
{
    /* locals */
    int i, ier;
    double w[4], summ, z;
    double a, b, xmin, xmax, ymin, ymax, dx, dy, dxx, dyy;
    char sub[24]="x_interp_map_nodes";

    xmin=UNDEF_MAP;
    xmax=-1*UNDEF_MAP;
    ymin=UNDEF_MAP;
    ymax=-1*UNDEF_MAP;

    z=0.0;

    xtgverbose(debug);

    xtg_speak(sub,3,"Entering routine");

    xtg_speak(sub,3,"INPUT: x y  %10.3f %10.3f", x, y);
    for (i=0; i<4; i++) {
	if (debug >=3) {
	    xtg_speak(sub,3,"%d x_v y_v z_v %10.3f %10.3f %10.3f",
		      i,x_v[i],y_v[i],z_v[i]);
	}
	if (x_v[i] < xmin) xmin =  x_v[i];
	if (y_v[i] < ymin) ymin =  y_v[i];
	if (x_v[i] > xmax) xmax =  x_v[i];
	if (y_v[i] > ymax) ymax =  y_v[i];
    }

    /*
     * ########################################################################
     * Some checks
     * ########################################################################
     */
    if ( x < xmin || x > xmax || y < ymin || y > ymax) {
	xtg_warn(sub,1,"Invalid input; x and y out of bound");
	xtg_warn(sub,1,"X= %10.2f X0= %10.2f X1= %10.2f Y= %10.2f Y0= %10.2f Y2= %10.2f",
		 x, x_v[0], x_v[1], y, y_v[0], y_v[2]);
	return UNDEF_MAP;
    }


    /* assume that all nodes are defined (this need to be better...?) */
    for (i=0;i<=3;i++) {
	if (z_v[i] > UNDEF_MAP_LIMIT) {
	return UNDEF_MAP;
	}
    }


    /* DEBUG4 section ++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
    if (debug > 3) {
	xtg_speak(sub,4,"Point X Y  is %f %f", x, y);
	for (i=0; i<4; i++) {
	    xtg_speak(sub,4,"X Y Z [%d] %f %f %f \n",
		      i, x_v[i], y_v[i], z_v[i]);
	}
    }
    /* DEBUG4 output end +++++++++++++++++++++++++++++++++++++++++++++++++++ */

    /*
     * ########################################################################
     * Method 1 use JCR home made interpolation
     * ########################################################################
     */
    if (method==1) {

	/*
	 * OK, we are inside mapnodes. Now I need a simple weight
	 * |---------------|
	 * |               | Use a quadratic inverse formula
	 * |           *   | for weights w[]
	 * |---------------|
	 */

	/* compute Eucledian(?) lengths */
	for (i=0; i<4; i++) {
	    w[i]=sqrt(pow((x-x_v[i]),2) + pow((y-y_v[i]),2));
	    /* DEBUG +++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
	    if (debug > 3) xtg_speak(sub,4,"W [%d] %f\n", i, w[i]);
	    /* DEBUG end +++++++++++++++++++++++++++++++++++++++++++++++++++ */
	}
	summ=0.0;
	for (i=0; i<4; i++) {
	    if (w[i]>0.00001) {
		w[i]=1.0/w[i];
	    }
	    else{
		w[i]=10000000.0;
	    }
	    summ=summ+w[i];
	}
	/* this should scale weights so that summ w[i] = 1.0 */
	for (i=0; i<4; i++) {
	    w[i]=w[i]/summ;
	}


	/*
	 * Find the z value node, weighted on distance
	 * UNDEF map nodes are given 0.0 in weight
	 */
	z=0.0;
	for (i=0; i<4; i++) {
/*	    if (z_v[i] <= UNDEF_MAP+2*FLOATEPS) w[i]=0.0; */
	    if (z_v[i] > UNDEF_MAP_LIMIT) w[i]=0.0;
	    z=z+z_v[i]*w[i];
	}
    }
    /*
     * ########################################################################
     * Method 2 bilinear formula (elegant...)
     * ########################################################################
     */
    else if (method==2) {

	/*
	 * Use a bilinear formula (from Tor Barkve)
	 * |---------------|
	 * |   *           | z=z1 + A(z2-z1) + B(z3-z1) + AB(z4+z1-z3-z2)
	 * |---------------| A=x/Dx   and   B=y/Dy
	 */

	a=(x-x_v[0])/(x_v[1]-x_v[0]);
	b=(y-y_v[0])/(y_v[2]-y_v[0]);

	z = z_v[0] + a*(z_v[1]-z_v[0]) + b*(z_v[2]-z_v[0])
	    + a*b*(z_v[3]+z_v[0]-z_v[2]-z_v[1]);
    }

    else if (method==3) {

	/*
	 * Use a bilinear formula, but with rotation of grid possible,
	 * hence deltas are computed instead (still assume regular map)
	 *
	 * 2               3
 	 * |---------------|
	 * |   *           |
	 * |---------------|
	 * 0               1
	 */


	dx = sqrt(pow(x_v[1]-x_v[0],2) + pow(y_v[1]-y_v[0],2));
	dy = sqrt(pow(x_v[2]-x_v[0],2) + pow(y_v[2]-y_v[0],2));

	/* find normal X distance from edge to point  */
	ier = x_point_line_dist(x_v[0], y_v[0], 0.0,
				x_v[2], y_v[2], 0.0,
				x, y, 0.0,
				&dxx, 0, 0, debug);

	if (ier == 2) dxx=0.0;
	if (ier == 1 || ier == 3) return(UNDEF_MAP);

	/* find normal Y distance from edge to point  */
	ier = x_point_line_dist(x_v[0], y_v[0], 0.0,
				x_v[1], y_v[1], 0.0,
				x, y, 0.0,
				&dyy, 0, 0, debug);

	if (ier == 2) dyy=0.0;
	if (ier == 1 || ier == 3) return(UNDEF_MAP);


	a=dxx/dx;
	b=dyy/dy;

	if (debug>2) {
	    xtg_speak(sub,3,"dxx and dx %f %f", dxx, dx);
	    xtg_speak(sub,3,"dyy and dy %f %f", dyy, dy);
	    xtg_speak(sub,3,"a and b is %f %f", a, b);
	}

	if (a > 1 || b > 1) {
	    xtg_warn(sub, 0, "Something is wrong, method 3 in %s", sub);
	    xtg_warn(sub, 0, "a = %f  and b = %f", a, b);
	    xtg_warn(sub, 0, "RESET!");
            if (a > 1) a=1;
            if (b > 1) b=1;
	}

	z = z_v[0] + a*(z_v[1]-z_v[0]) + b*(z_v[2]-z_v[0])
	    + a*b*(z_v[3]+z_v[0]-z_v[2]-z_v[1]);
    }

    return z;

}
