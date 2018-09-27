/*
 ******************************************************************************
 *
 * Find XY origin values from a given XY and IJ value (and incs + rot)
 *
 ******************************************************************************
 */

#include <math.h>
#include "libxtg.h"
#include "libxtg_.h"

/*
 ******************************************************************************
 *
 * NAME:
 *    surf_xyori_from_ij.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
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
 ******************************************************************************
 */


#include "libxtg.h"
#include "libxtg_.h"

int surf_xyori_from_ij(
                       int i,
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
                       int flag,
                       int debug
                       )
{
    /* locals */
    char s[24]="surf_xyori_from_ij";
    double angle, xdist, ydist, dist, beta, gamma, dxrot, dyrot;


    xtgverbose(debug);
    xtg_speak(s,3,"Entering routine %s", s);

    if (i < 1 || i > nx || j < 1 || j > ny) {
        xtg_error(s,"%s: Error in I J spec: out of bounds %d %d"
                  " (%d %d)", s, i, j, nx, ny);
        return -1;
    }

    if (i==1 && j==1) {
        *xori = x;
        *yori = y;
        return(0);
    }

    if (debug>2) xtg_speak(s,3,"YFLIP is %d", yflip);

    yinc = yinc * yflip;

    /* cube rotation: this should be the usual angle, anti-clock from x axis */
    angle = (rot_deg) *PI / 180.0;  /* radians, positive */

    xdist = xinc * (i - 1);
    ydist = yinc * (j - 1);

    /* distance of point from "origo" */
    dist = sqrt(xdist * xdist + ydist * ydist);

    /* beta is the angle of line from origo to point, if nonrotated system */
    xtg_speak(s,3,"XDIST and YDIST and DIST %6.2f %6.2f  %6.2f",
              xdist,ydist,dist);

    beta = acos(xdist / dist);

    if (debug>2) {
	   xtg_speak(s, 3, "Angles are %6.2f  %6.2f",
                     angle * 180 / PI, beta * 180 / PI);
    }

    /* secure that angle is in right mode */
    /* if (xdist<0 && ydist<0)  beta=2*PI - beta; */
    /* if (xdist>=0 && ydist<0) beta=PI + beta; */

    if (beta < 0 || beta > PI/2.0) {
	xtg_error(s, "FATAL: Beta is wrong, call JRIV...\n");
	return(-9);
    }

    gamma = angle + yflip * beta; /* the difference in rotated coord system */

    dxrot = dist * cos(gamma);
    dyrot = dist * sin(gamma);

    if (debug>2) {
	xtg_speak(s,3,"DXROOT DYROOT %f %f", dxrot, dyrot);
    }

    *xori = x - dxrot;
    *yori = y - dyrot;

    xtg_speak(s, 2, "Return from %s", s);
    return(0);
}
