/*
 * ############################################################################
 * Find mean of N angles (degrees)
 * CF https://rosettacode.org/wiki/Averages/Mean_angle#C
 * ############################################################################
 */

#include <math.h>
#include "libxtg.h"
#include "libxtg_.h"


double x_avg_angles(double *angles, int nsize)
{
    double y_part = 0, x_part = 0, angle;
    int i;

    for (i = 0; i < nsize; i++) {
        x_part += cos(angles[i] * M_PI / 180);
        y_part += sin(angles[i] * M_PI / 180);
    }

    angle = atan2 (y_part / nsize, x_part / nsize) * 180 / M_PI;

    /* keep in [0, 360 > */
    while (angle < 0.0) angle += 360;
    while (angle >= 360.0) angle -= 360;
    return angle;

}
