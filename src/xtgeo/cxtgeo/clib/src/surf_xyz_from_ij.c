/*
 ******************************************************************************
 *
 * Find XYZ ("Z" is any value in general) from surface given IJ
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
 *    surf_xyz_from_ij.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *     Given a map node IJ, the X Y "ZVALUE" is computed
 *
 * ARGUMENTS:
 *    i, j           i     col/row node
 *    x, y, z        o     Output
 *    xori           i     X origin coordinate
 *    xinc           i     X increment
 *    yori           i     Y origin coordinate
 *    yinc           i     Y increment
 *    nx, ny         i     Dimensions
 *    yflip          i     YFLIP indicator 1 or -1
 *    rot            i     Rotation (degrees, from X axis, anti-clock)
 *    p_map_v        i     map array
 *    nn             i     Array length
 *    flag           i     Options flag; 1 if z value (and map) is discarded)
 *
 * RETURNS:
 *    Function: 0: upon success. If problems <> 0:.
 *    X Y Z pointers updated
 *
 * TODO/ISSUES/BUGS:
 *    yflip handling?
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ******************************************************************************
 */


#include "libxtg.h"
#include "libxtg_.h"

int surf_xyz_from_ij(
                     int i,
                     int j,
                     double *x,
                     double *y,
                     double *z,
                     double xori,
                     double xinc,
                     double yori,
                     double yinc,
                     int nx,
                     int ny,
                     int yflip,
                     double rot_deg,
                     double *p_map_v,
                     long nn,
                     int flag
                     )
{
    /* locals */
    double   angle, xdist, ydist, dist, beta, gamma, dxrot, dyrot;
    int      ic;


    if (i<1 || i>nx || j<1 || j>ny) {
        if (i == 0) i=1;
        if (i == nx + 1) i = nx;
        if (j == 0) j = 1;
        if (j == ny + 1) j = ny;

        /* retest if more severe and return -1 if case*/
        if (i<1 || i>nx || j<1 || j>ny) {
            return -1;
        }
    }

    if (flag == 0) {
        ic = x_ijk2ic(i,j,1,nx,ny,1,0);  /* C order */
        *z = p_map_v[ic];
    }
    else{
        *z = 999.00;
    }


    if (i==1 && j==1) {
        *x = xori;
        *y = yori;
        return(0);
    }

    yinc = yinc * yflip;

    /* cube rotation: this should be the usual angle, anti-clock from x axis */
    angle = (rot_deg) * PI / 180.0;  /* radians, positive */

    xdist = xinc * (i - 1);
    ydist = yinc * (j - 1);

    /* distance of point from "origo" */
    dist = sqrt(xdist * xdist + ydist * ydist);

    beta = acos(xdist / dist);

    /* secure that angle is in right mode */
    /* if (xdist<0 && ydist<0)  beta=2*PI - beta; */
    /* if (xdist>=0 && ydist<0) beta=PI + beta; */

    if (beta < 0 || beta > PI/2.0) {
	return(-9);
    }

    gamma = angle + yflip * beta; /* the difference in rotated coord system */

    dxrot = dist * cos(gamma);
    dyrot = dist * sin(gamma);

    *x = xori + dxrot;
    *y = yori + dyrot;

    return(0);
}
