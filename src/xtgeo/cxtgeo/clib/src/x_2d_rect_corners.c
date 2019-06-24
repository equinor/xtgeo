/*
 ***************************************************************************************
 *
 * Corners ofa 2D rotated rectangle
 *
 ***************************************************************************************
 */
#include <math.h>
#include "libxtg.h"
#include "libxtg_.h"

/*
 ***************************************************************************************
 *
 * NAME:
 *    x_2d_rect_corners.c
 *
 * DESCRIPTION:
 *    Find corners of a 2D rectangle, given midpoint, rotation and length
 *    Cf. https://gamedev.stackexchange.com/questions/86755/\
 *    how-to-calculate-corner-positions-marks-of-a-rotated-tilted-rectangle
 *
 * ARGUMENTS:
 *    x, y           i     Mid point X, Y
 *    xinc, yinc           Increments
 *    rotation       i     Angle in degrees, anticlock from X axis
 *    result         o     an array (x, y, ...starting from relative upper left
 *                         in clockwise direction
 *    debug          i     Debug level
 *
 * RETURNS:
 *    Void
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ***************************************************************************************
 */
void x_2d_rect_corners(double x, double y, double xinc, double yinc, double rot,
                      double result[8], int debug)
{
    xtgverbose(debug);

    rot = rot * PI / 180.0;

    double cv = cos(rot);
    double sv = sin(rot);

    double r1x = -0.5 * xinc * cv - 0.5 * yinc * sv;
    double r1y = -0.5 * xinc * sv + 0.5 * yinc * cv;
    double r2x = 0.5 * xinc * cv - 0.5 * yinc * sv;
    double r2y = 0.5 * xinc * sv + 0.5 * yinc * cv;

    result[0] = x + r1x;
    result[1] = y + r1y;
    result[2] = x + r2x;
    result[3] = y + r2y;
    result[4] = x - r1x;
    result[5] = y - r1y;
    result[6] = x - r2x;
    result[7] = y - r2y;
}
