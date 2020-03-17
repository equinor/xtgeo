/*
 ***************************************************************************************
 *
 * NAME:
 *    surf_xy_as_values.c
 *
 *(S):
 *
 *
 * DESCRIPTION:
 *    Returns the X and Y coordinates as map values
 *
 * ARGUMENTS:
 *    xori       i      X origin
 *    xinc       i      X increment
 *    yori       i      Y origin
 *    yinc       i      Y increment (yflip -1 is assumed if yinc < 0)
 *    nx         i      NX (columns)
 *    ny         i      NY (rows)
 *    rot_deg    i      rotation
 *    p_x_v      o      pointer to X values (must be allocated in caller)
 *    nn1        i      length of prev array (for allocation from SWIG)
 *    p_y_v      9      pointer to Y values (must be allocated in caller)
 *    nn2        i      length of prev array (for allocation from SWIG)
 *    flag       i      Flag for options
 *    debug      i      Debug flag
 *
 * RETURNS:
 *    Int function, returns 0 upon success + updated X and Y pointers
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ***************************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"
#include "logger.h"
#include <math.h>

int
surf_xy_as_values(double xori,
                  double xinc,
                  double yori,
                  double yinc,
                  int nx,
                  int ny,
                  double rot_deg,
                  double *p_x_v,
                  long nn1,
                  double *p_y_v,
                  long nn2,
                  int flag)
{
    /* locals */
    double angle, xdist, ydist, dist, beta, gamma, dxrot = 0.0, dyrot = 0.0;
    int i, j, ib, yflip;

    if (nx * ny != nn1 || nn1 != nn2) {
        logger_error(LI, FI, FU, "Error? in length nn1 vs nx*ny or nn1 vs nn2 in %s",
                     FU);
    }

    yflip = 1;
    if (yinc < 0.0) {
        yflip = -1;
        yinc = yinc * yflip;
    }

    /* surf rotation: this should be the usual angle, anti-clock from x axis */
    angle = (rot_deg)*PI / 180.0; /* radians, positive */

    for (i = 1; i <= nx; i++) {
        for (j = 1; j <= ny; j++) {

            ib = x_ijk2ic(i, j, 1, nx, ny, 1, 0); /* C order */

            if (i == 1 && j == 1) {

                p_x_v[ib] = xori + dxrot;
                p_y_v[ib] = yori + dyrot;

            } else {

                xdist = xinc * (i - 1);
                ydist = yinc * (j - 1);

                /* distance of point from "origo" */
                dist = sqrt(xdist * xdist + ydist * ydist);

                /* beta is the angle of line from origo to point, assuming
                   nonrotated system */

                if (debug > 2) {
                    xtg_speak(s, 3, "XDIST / YDIST / DIST %6.2f %6.2f  %6.2f", xdist,
                              ydist, dist);
                }

                beta = acos(xdist / dist);

                if (debug > 2) {
                    xtg_speak(s, 3, "Angles are %6.2f  %6.2f", angle * 180 / PI,
                              beta * 180 / PI);
                }

                /* secure that angle is in right mode */
                /* if (xdist<0 && ydist<0)  beta=2*PI - beta; */
                /* if (xdist>=0 && ydist<0) beta=PI + beta; */

                if (beta < 0 || beta > PI / 2.0) {
                    xtg_error(s, "Beta is wrong, call JRIV...\n");
                    return (-1);
                }

                /* the difference in rotated coord system */
                gamma = beta * yflip + angle;

                dxrot = dist * cos(gamma);
                dyrot = dist * sin(gamma);

                p_x_v[ib] = xori + dxrot;
                p_y_v[ib] = yori + dyrot;
            }
        }
    }
    return EXIT_SUCCESS;
}
