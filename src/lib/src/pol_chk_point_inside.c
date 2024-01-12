/*
 ***************************************************************************************
 *
 * NAME:
 *    pol_check_point_inside.c
 *
 * DESCRIPTION:
 *    Check if a point is inside a polygon
 *
 * ARGUMENTS:
 *    x, y               i     Point
 *    xvertices, yv..   i/o    arrays
 *    np                 i     Number of points for polygon
 *
 * RETURNS:
 *    If >= 1, point is inside, if 0 outside, if -1 outside
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    CF XTGeo's LICENSE
 ***************************************************************************************
 */
#include <math.h>

#include <xtgeo/xtgeo.h>

#include "logger.h"

int
pol_chk_point_inside(double x, double y, double *xvertices, double *yvertices, int np)

{
    double CNULL, CEN, PIH, TOPI, XEPS, DTMP;
    int i;

    /*
     *----------------------------------------------------------------------------------
     * Constants
     *----------------------------------------------------------------------------------
     */

    CNULL = 0.0;
    CEN = 1.0;
    PIH = asin(CEN);
    TOPI = 4.0 * PIH;
    DTMP = (double)np;
    XEPS = sqrt(DTMP) * 1.0e-5; /* 1e-3? works better than e-09 in pp */

    /*
     *----------------------------------------------------------------------------------
     * Check
     *----------------------------------------------------------------------------------
     */

    /* check first vs last point, and force close if small */
    double xdiff = fabs(xvertices[0] - xvertices[np - 1]);
    double ydiff = fabs(yvertices[0] - yvertices[np - 1]);

    if (xdiff < FLOATEPS && ydiff < FLOATEPS) {
        xvertices[np - 1] = xvertices[0];
        yvertices[np - 1] = yvertices[0];
    } else {
        logger_warn(LI, FI, FU, "Not a closed polygon, return -9");
        int n;
        for (n = 0; n < np; n++) {
            logger_warn(LI, FI, FU, "Point no %d: %lf %lf", n, xvertices[n],
                        yvertices[n]);
        }
        return -9;
    }

    /*
     *-------------------------------------------------------------------------
     * Loop over all corners (edges)
     *-------------------------------------------------------------------------
     */
    double vinsum = CNULL;
    double x2 = xvertices[np - 1] - x;
    double y2 = yvertices[np - 1] - y;

    for (i = 0; i < np; i++) {
        /* differences and norms */
        double x1 = x2;
        double y1 = y2;
        x2 = xvertices[i] - x;
        y2 = yvertices[i] - y;
        double an1 = sqrt(x1 * x1 + y1 * y1);
        double an2 = sqrt(x2 * x2 + y2 * y2);
        double an = an1 * an2;

        if (an == CNULL) {
            /* points is on a corner */
            return (1);
        }
        /* cross-product and dot-product */
        double xp = x1 * y2 - x2 * y1;
        double pp = x1 * x2 + y2 * y1;

        /* compute scalar value of angle: 0 <= vin <= pi */
        double cosv = pp / an;
        if (cosv > CEN)
            cosv = CEN;
        if (cosv < -1 * CEN)
            cosv = -1 * CEN;
        double vin = acos(cosv);

        if (xp == CNULL) {
            if (vin >= PIH) {
                /* vin==pi -> point on edge */
                return (1);
            } else {
                vin = CNULL;
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
    if (fabs(vinsum - TOPI) <= XEPS) {
        return 2;
    } else if (vinsum <= XEPS) {
        return 0;
    } else {
        return -1;
    }
}
