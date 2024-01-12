/*
 ***************************************************************************************
 *
 * NAME:
 *    x_diff_angle.c
 *
 * DESCRIPTION:
 *    Finds the smallest difference between two angles, taking the circle into
 *    account. This routine thinks clockwise direction
 *
 *    Examples (degrees): Requested=30, Actual=40, result shall be -10
 *                        Requested=360, Actual=340, result shall be 20
 *                        Requested=360, Actual=170, result shall be -170
 *
 *    https://rosettacode.org/wiki/Angle_difference_between_two_bearings#C++
 *
 * ARGUMENTS:
 *    ang1           Requested angle
 *    ang2           Actual angle
 *    option         1 if degrees, otherwise radians
 *
 * RETURNS:
 *    Angle difference (with sign): Requested - Actual
 *    i.e.: Actual + Diff = Requested
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    CF XTGeo's LICENSE
 ***************************************************************************************
 */
#include <math.h>

#include <xtgeo/xtgeo.h>

double
x_diff_angle(double ang1, double ang2, int option)
{
    double full = 2 * M_PI, half = M_PI;
    double diff;

    if (option == 1) {
        full = 360.0;
        half = 180.0;
    }

    diff = fmod(ang1 - ang2, full);
    if (diff < -half)
        diff += full;
    if (diff > half)
        diff -= full;

    return diff;
}
