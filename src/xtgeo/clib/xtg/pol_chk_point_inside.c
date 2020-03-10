/*
 ***************************************************************************************
 *
 * NAME:
 *    some.c
 *
 * DESCRIPTION:
 *    Some
 *
 * ARGUMENTS:
 *    x, y             i     Point
 *    p_xp_v, ..yp_v  i/o     arrays
 *    np               i     Coordinate vector (with numpy dimensions)
 *
 * RETURNS:
 *    Status, EXIT_FAILURE or EXIT_SUCCESS
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    CF XTGeo's LICENSE
 ***************************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"
#include "logger.h"
#include <math.h>

int
pol_chk_point_inside(double x, double y, double *p_xp_v, double *p_yp_v, int np)

{
    double cnull, cen, pih, topi, eps;
    double x1, x2, y1, y2, vin, vinsum, an, an1, an2, xp, pp;
    double cosv, dtmp, xdiff, ydiff;
    int i;
    char s[24] = "pol_chk_point_inside";

    /*
     *-------------------------------------------------------------------------
     * Constants
     *-------------------------------------------------------------------------
     */

    cnull = 0.0;
    cen = 1.0;
    pih = asin(cen);
    topi = 4.0 * pih;
    dtmp = np;
    eps = sqrt(dtmp) * 1.0e-3; /*works better than e-09 in pp */

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
    } else {
        xtg_warn(s, 2, "Not a closed polygon, return -9");
        return -9;
    }

    /*
     *-------------------------------------------------------------------------
     * Loop over all corners (edges)
     *-------------------------------------------------------------------------
     */
    vinsum = cnull;
    x2 = p_xp_v[np - 1] - x;
    y2 = p_yp_v[np - 1] - y;

    for (i = 0; i < np; i++) {
        /* differences and norms */
        x1 = x2;
        y1 = y2;
        x2 = p_xp_v[i] - x;
        y2 = p_yp_v[i] - y;
        an1 = sqrt(x1 * x1 + y1 * y1);
        an2 = sqrt(x2 * x2 + y2 * y2);
        an = an1 * an2;

        if (an == cnull) {
            /* points is on a corner */
            return (1);
        }
        /* cross-product and dot-product */
        xp = x1 * y2 - x2 * y1;
        pp = x1 * x2 + y2 * y1;

        /* compute scalar value of angle: 0 <= vin <= pi */
        cosv = pp / an;
        if (cosv > cen)
            cosv = cen;
        if (cosv < -1 * cen)
            cosv = -1 * cen;
        vin = acos(cosv);

        if (xp == cnull) {
            if (vin >= pih) {
                /* vin==pi -> point on edge */
                return (1);
            } else {
                vin = cnull;
            }
        } else {
            /* angle use same +- sign as cross-product (implement Fortran SIGN)*/
            if (xp >= 0.0) {
                vin = fabs(vin);
            } else {
                vin = -1 * fabs(vin);
            }
        }
        vinsum = vinsum + vin;
    }
    vinsum = fabs(vinsum);

    /* determine inside or... */
    if (fabs(vinsum - topi) <= eps) {
        return (2);
    } else if (vinsum <= eps) {
        return (0);
    } else {
        return (-1);
    }
}
