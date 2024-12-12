/*
 ***************************************************************************************
 *
 * NAME:
 *    x_linint3d.c
 *
 * DESCRIPTION:
 *    Assume two points x0, y0, z0 and x1, y1, z1. Given zp, find xp and yp. Typical
 *    interpolation in COORDS when ZCORN is known. Third (Z) value must separated
 *    in space.
 *
 * ARGUMENTS:
 *    p0             i     Array 3 values x y z
 *    p1             i     Array 3 values x y z
 *    zp             i     known value
 *    xp             o     X value to be estimated
 *    yp             o     Y value to be estimated
 *
 * RETURNS:
 *    EXIT_SUCCESS if OK. Estimated yp and xp.
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ***************************************************************************************
 */
#include <math.h>
#include <stdlib.h>
#include <xtgeo/xtgeo.h>
#include "common.h"
#include "logger.h"

int
x_linint3d(double *p0, double *p1, double zp, double *xp, double *yp)
{

    if (fabs(p1[2] - p0[2]) < FLOATEPS) {
        return EXIT_FAILURE;
    }

    double ratio = (zp - p0[2]) / (p1[2] - p0[2]);

    *xp = p0[0] + ratio * (p1[0] - p0[0]);
    *yp = p0[1] + ratio * (p1[1] - p0[1]);

    return EXIT_SUCCESS;
}
