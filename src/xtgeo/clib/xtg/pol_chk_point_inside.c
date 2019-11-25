#include <math.h>
#include "libxtg.h"
#include "libxtg_.h"

/*
 *=============================================================================
 * Check if a point is within a polygon in 2D. The poly has to be organizated
 * cyclic, either clockwise or opposite.
 * Strange forms (stars etc) should be OK.
 * This routine is adapted from numlib (kvppgs.f)
 * Return number:
 * 2 if point inside polygon
 * 1 if on edge
 * 0 outside
 * -1 undetermined
 * -9 Polygon is not closed
 *=============================================================================
 * Notice:
 * Based on polys_chk_point_inside, but limited to one polygon. This shall
 * be consistent with the Polygon class. Also note that UNDEF values
 * shall not exist. If the last value differs from the first, an error
 * is triggered.
 *=============================================================================
 */

int pol_chk_point_inside(
			 double x,
			 double y,
			 double *p_xp_v,
			 double *p_yp_v,
			 int    np,       // number of points, counted from 1
			 int    debug
			 )

{
    double cnull, cen, pih, topi, eps;
    double x1 ,x2, y1, y2, vin, vinsum, an, an1, an2, xp, pp;
    double cosv, dtmp, xdiff, ydiff;
    int i;
    char s[24] = "pol_chk_point_inside";

    xtgverbose(debug);
    /*
     *-------------------------------------------------------------------------
     * Constants
     *-------------------------------------------------------------------------
     */

    cnull=0.0;
    cen=1.0;
    pih=asin(cen);
    topi=4.0*pih;
    dtmp=np;
    eps=sqrt(dtmp)*1.0e-3; /*works better than e-09 in pp */

    /*
     *-------------------------------------------------------------------------
     * Check
     *-------------------------------------------------------------------------
     */


    /* check first vs last point, and force close if small */
    xdiff = fabs(p_xp_v[0] - p_xp_v[np - 1]);
    ydiff = fabs(p_yp_v[0] - p_yp_v[np - 1]);

    if (xdiff < FLOATEPS && ydiff < FLOATEPS) {
        p_xp_v[np - 1] = p_xp_v[0];
        p_yp_v[np - 1] = p_yp_v[0];
    }
    else{
	xtg_warn(s, 2, "Not a closed polygon, return -9");
        return -9;
    }

    /*
     *-------------------------------------------------------------------------
     * Loop over all corners (edges)
     *-------------------------------------------------------------------------
     */
    vinsum=cnull;
    x2=p_xp_v[np-1]-x;
    y2=p_yp_v[np-1]-y;

    for (i=0;i<np;i++) {
	if (debug > 3) xtg_speak(s, 4, "Polygon corners is %f %f",
                                 p_xp_v[i], p_yp_v[i]);
	/* differences and norms */
	x1=x2;
	y1=y2;
	x2=p_xp_v[i]-x;
	y2=p_yp_v[i]-y;
	an1=sqrt(x1*x1 + y1*y1);
	an2=sqrt(x2*x2 + y2*y2);
	an=an1*an2;

	if (an == cnull) {
	    /* points is on a corner */
	    return(1);
	}
	/* cross-product and dot-product */
	xp = x1*y2 - x2*y1;
	pp = x1*x2 + y2*y1;

	/* compute scalar value of angle: 0 <= vin <= pi */
	cosv=pp/an;
	if (cosv > cen) cosv=cen;
	if (cosv < -1*cen) cosv = -1*cen;
	vin=acos(cosv);

	if (xp == cnull) {
	    if (vin >= pih) {
		/* vin==pi -> point on edge */
		return(1);
	    }
	    else{
		vin=cnull;
	    }
	}
	else{
	    /* angle use same +- sign as cross-product (implement Fortran SIGN)*/
	    if (xp >= 0.0) {
		vin=fabs(vin);
	    }
	    else{
		vin=-1*fabs(vin);
	    }

	}
	vinsum=vinsum+vin;

    }
    vinsum=fabs(vinsum);

    /* determine inside or... */
    if (fabs(vinsum-topi) <= eps) {
	return(2);
    }
    else if (vinsum <= eps) {
	return(0);
    }
    else{
	return(-1);
    }
}
