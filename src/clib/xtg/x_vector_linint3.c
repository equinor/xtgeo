
/*
 ***************************************************************************************
 *
 * NAME:
 *    x_vector_linint3.c
 *
 * DESCRIPTION:
 *    Assume two variables and 3 values for each: x0 <= x1 <= x2,
 *    and y: y0 <= y1 <= y2
 *    For x, all values are known, but for y, y1 is unknown. The purpose is to return
 *    the interpolated value y1 based on x1
 *
 * ARGUMENTS:
 *    x0             i     X start point 0
 *    x1             i     X start point 1
 *    x2             i     X start point 2
 *    y0             i     Y start point 0
 *    y2             i     Y start point 2
 *
 * RETURNS:
 *    Estimated y1
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ***************************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"
#include "logger.h"

double
x_vector_linint3(double x0, double x1, double x2, double y0, double y2)
{

    if (fabs(x2 - x0) < FLOATEPS)
        return y0;

    if (x1 < x0 || x1 > x2 || x2 < x0) {
        logger_critical(LI, FI, FU, "Input values wrong for %s", FU);
    }

    double xratio = (x1 - x0) / (x2 - x0);

    double y1 = y0 + xratio * (y2 - y0);

    return y1;
}
