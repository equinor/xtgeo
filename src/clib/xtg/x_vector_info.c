/*
 ***************************************************************************************
 *
 * NAME:
 *    x_vector_info2.c
 *
 * DESCRIPTION:
 *    Take 2 points in XY space and compute length and azimuth or normal angle
 *    Angles shall be in range 0-360 degrees (no negative angles)
 *
 * ARGUMENTS:
 *    x1 ... y2        i     Points
 *    vlen             o     Length (2D, XY space)
 *    xangle_radian    o     Angle, radians
 *    xangle_degrees   o     Angle, degrees
 *    option           i     0: azimuth returned, 1: angle (aka school) is returned
 *
 * RETURNS:
 *    Option 0, AZIMUTH is returned (clockwise, releative to North)
 *    Option 1, ANGLE is returned (counter clockwise, relative to East)
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

void
x_vector_info2(double x1,
               double x2,
               double y1,
               double y2,
               double *vlen,
               double *xangle_radian,
               double *xangle_degrees,
               int option)
{
    /* locals */
    double azi, deg;

    /*
     * ------------------------------------------------------------------------
     * Some checks
     * ------------------------------------------------------------------------
     */

    if (x1 == x2 && y1 == y2) {
        *vlen = 0.000001;
        *xangle_radian = 0.0;
        *xangle_degrees = 0.0;
        return;
    }

    /*
     * ------------------------------------------------------------------------
     * Compute
     * ------------------------------------------------------------------------
     */

    *vlen = sqrt(powf(x2 - x1, 2) + powf(y2 - y1, 2));

    if ((x2 - x1) > 0.00001 || (x2 - x1) < -0.00001) {

        deg = atan((y2 - y1) / (x2 - x1));
        /* western quadrant */
        if (x2 > x1) {
            azi = PI / 2 - deg;
        }
        /* eastern quadrant */
        else {
            deg = deg + PI;
            azi = 2 * PI + PI / 2 - deg;
        }

    } else {
        if (y2 < y1) {
            azi = PI;
            deg = -PI / 2.0;
        } else {
            azi = 0;
            deg = PI / 2;
        }
    }

    if (azi < 0)
        azi = azi + 2 * PI;
    if (azi > 2 * PI)
        azi = azi - 2 * PI;

    if (deg < 0)
        deg = deg + 2 * PI;
    if (deg > 2 * PI)
        deg = deg - 2 * PI;

    *xangle_radian = azi;

    /* normal school angle */
    if (option == 1) {
        *xangle_radian = deg;
    }

    *xangle_degrees = *(xangle_radian)*180 / PI;
}

/*
 ***************************************************************************************
 *
 * NAME:
 *    x_vector_info2.c
 *
 * DESCRIPTION:
 *    Length of line in 3D between P1 and P2
 *
 ***************************************************************************************
 */

double
x_vector_len3d(double x1, double x2, double y1, double y2, double z1, double z2)
{
    double vlen;

    if (x1 == x2 && y1 == y2 && z1 == z2) {
        vlen = 10E-20;
    } else {
        vlen = sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2) + pow(z2 - z1, 2));
    }
    return vlen;
}

double
x_vector_len3dx(double x1, double y1, double z1, double x2, double y2, double z2)
// aa but different order of input items
{
    double vlen;

    if (x1 == x2 && y1 == y2 && z1 == z2) {
        vlen = 10E-20;
    } else {
        vlen = sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2) + pow(z2 - z1, 2));
    }
    return vlen;
}
