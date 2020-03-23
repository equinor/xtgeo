
/*
 ***************************************************************************************
 *
 * NAME:
 *    surf_xyori_from_ij.c
 *
 * DESCRIPTION:
 *     Given a map node IJ and known XY, the xori and yori is computed.
 *
 * ARGUMENTS:
 *    i, j           i     col/row node (NB base is 1, not 0)
 *    x, y           i     Input X Y at I J
 *    xori           o     X origin coordinate
 *    xinc           i     X increment
 *    yori           o     Y origin coordinate
 *    yinc           i     Y increment
 *    nx, ny         i     Dimensions
 *    yflip          i     YFLIP indicator 1 or -1
 *    rot            i     Rotation (degrees, from X axis, anti-clock)
 *    debug          i     Debug level
 *
 * RETURNS:
 *    Function: 0: upon success. If problems <> 0:.
 *    X Y ..ori pointers are updated
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    See XTGeo lisence
 ***************************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"
#include "logger.h"
#include <math.h>

int
surf_xyori_from_ij(int i,
                   int j,
                   double x,
                   double y,
                   double *xori,
                   double xinc,
                   double *yori,
                   double yinc,
                   int nx,
                   int ny,
                   int yflip,
                   double rot_deg,
                   int flag)
{
    /* locals */
    double angle, xdist, ydist, dist, beta, gamma, dxrot, dyrot;

    if (i < 1 || i > nx || j < 1 || j > ny) {
        logger_error(LI, FI, FU,
                     "%s: Error in I J spec: out of bounds %d %d"
                     " (%d %d)",
                     FU, i, j, nx, ny);
        return -1;
    }

    if (i == 1 && j == 1) {
        *xori = x;
        *yori = y;
        return (0);
    }

    yinc = yinc * yflip;

    /* cube rotation: this should be the usual angle, anti-clock from x axis */
    angle = (rot_deg)*PI / 180.0; /* radians, positive */

    xdist = xinc * (i - 1);
    ydist = yinc * (j - 1);

    /* distance of point from "origo" */
    dist = sqrt(xdist * xdist + ydist * ydist);

    /* beta is the angle of line from origo to point, if nonrotated system */
    beta = acos(xdist / dist);

    /* secure that angle is in right mode */
    /* if (xdist<0 && ydist<0)  beta=2*PI - beta; */
    /* if (xdist>=0 && ydist<0) beta=PI + beta; */

    if (beta < 0 || beta > PI / 2.0) {
        logger_error(LI, FI, FU, "Bug: Beta is wrong!");
        return (-9);
    }

    gamma = angle + yflip * beta; /* the difference in rotated coord system */

    dxrot = dist * cos(gamma);
    dyrot = dist * sin(gamma);

    *xori = x - dxrot;
    *yori = y - dyrot;

    return (0);
}
