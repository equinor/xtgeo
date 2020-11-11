
/*
 ******************************************************************************
 *
 * NAME:
 *    x_vector_linint2.c
 *
 *
 * DESCRIPTION:
 *    Give 2 points in space, do a "forecast" of a point with a given distance
 *    which shall be good for extrapolation, but also interpolation of the
 *    P0 and P1 have larger distance than "dist".
 *
 * ARGUMENTS:
 *    x0             i     X start point P0
 *    y0             i     Y start point P0
 *    z0             i     Z start point P0
 *    x1             i     X next point P1
 *    y1             i     Y next point P1
 *    z1             i     Z next point P1
 *    dist           i     Requested distance from P1
 *    xr             o     X Returned forecast of a point
 *    yr             o     Y Returned forecast of a point
 *    zr             o     Z Returned forecast of a point
 *    option         i     Options: 1 Extend in X+ with 1 unit if P0 and P1 are equal
 *                                  2 Extend in X- with 1 unit if P0 and P1 are equal
 *                                 11 Extend X+ with dist if P0 and P1 are equal
 *                                 12 Extend X- with dist if P0 and P1 are equal
 *
 * RETURNS:
 *    Function:  0: Upon success. If problems:
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ******************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"
#include <math.h>

int
x_vector_linint2(double x0,
                 double y0,
                 double z0,
                 double x1,
                 double y1,
                 double z1,
                 double dist,
                 double *xr,
                 double *yr,
                 double *zr,
                 int option)
{
    /* locals */
    double length, ux, uy, uz, x2, y2, z2;

    /*
     * ------------------------------------------------------------------------
     * Some checks and options processing
     * ------------------------------------------------------------------------
     */

    if (fabs(x1 - x0) < 1e-20 && fabs(y1 - y0) < 1e-20) {
        if (option == 0) {
            return -1;
        } else if (option == 1) {
            x1 = x1 + 1;
        } else if (option == 2) {
            x1 = x1 - 1;
        } else if (option == 11) {
            *xr = x1 + dist;
            *yr = y1;
            *zr = z1;
            return 0;
        } else if (option == 12) {
            *xr = x1 - dist;
            *yr = y1;
            *zr = z1;
            return 0;
        } else {
            return -99;
        }
    }

    /*
     * ------------------------------------------------------------------------
     * Compute, find the vector
     * ------------------------------------------------------------------------
     */

    length = sqrt(pow(x1 - x0, 2) + pow(y1 - y0, 2) + pow(z1 - z0, 2));

    if (length < 1e-22) {
        return -9;
    }

    ux = (x1 - x0) / length;
    uy = (y1 - y0) / length;
    uz = (z1 - z0) / length;

    x2 = x1 + ux * dist;
    y2 = y1 + uy * dist;
    z2 = z1 + uz * dist;

    *xr = x2;
    *yr = y2;
    *zr = z2;

    return 0;
}
