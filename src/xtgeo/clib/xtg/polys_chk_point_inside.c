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
 * JCR 17-FEB-2002
 *=============================================================================
 * Notice:
 * p_xp_v and p_yp_v may contain several polygons. It is important that this
 * routine uses a single polygon within the set. This is noted by using
 * np1 (start) and np2 (stop). Example:
 * 534107.203857 6739110.818115   0
 * 533039.560425 6739059.977051   1
 * 531971.916992 6738907.457275   2
 * 531674.519531 6739027.486572   3
 * 531356.117920 6739693.593750   4
 * -999.000000 -999.000000        5
 * 523865.425781 6751717.128906   6    NP1
 * 524242.802734 6752008.737305   7
 * 524932.494141 6752279.557617   8
 * 524143.175781 6750989.322266   9
 * 523992.173828 6751308.480469   10
 * 523865.425781 6751717.128906   11   NP2
 * -999.000000 -999.000000        12
 *=============================================================================
 */

int polys_chk_point_inside(
			   double x,
			   double y,
			   double *p_xp_v,
			   double *p_yp_v,
			   int    np1,
			   int    np2,
			   int    debug
			   )

{
    double cnull, cen, pih, topi, eps;
    double x1 ,x2, y1, y2, vin, vinsum, an, an1, an2, xp, pp;
    double cosv, dtmp, diffx, diffy;
    int    i;
    char  s[24]="polys_chk_point_inside";


    xtgverbose(debug);

    xtg_speak(s,2,"Entering routine %s", s);

    /*
     *-------------------------------------------------------------------------
     * Constants
     *-------------------------------------------------------------------------
     */

    cnull=0.0;
    cen=1.0;
    pih=asin(cen);
    topi=4.0*pih;
    dtmp=np2-np1+1;
    eps=sqrt(dtmp)*1.0e-3; /*works better than e-09 in pp */

    /*
     *-------------------------------------------------------------------------
     * Loop over all corners (edges)
     *-------------------------------------------------------------------------
     */

    diffx = fabs(p_xp_v[np1] - p_xp_v[np2]);
    diffy = fabs(p_yp_v[np1] - p_yp_v[np2]);

    if (diffx > 1e-10 || diffy > 1e-10) {
        /* polygon is not closed */
        return -9;
    }

    vinsum=cnull;
    x2=p_xp_v[np2]-x;
    y2=p_yp_v[np2]-y;

    for (i=np1;i<=np2;i++) {
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
