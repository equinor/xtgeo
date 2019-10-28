/*
****************************************************************************************
 *
 * NAME:
 *    surf_get_dist_values.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Given a point in space, and an azimuth, then compute map values as
 *    distance from the line perpendicular to the point and azimuth.
 *
 * ARGUMENTS:
 *    xori       i      X origin
 *    xinc       i      X increment
 *    yori       i      Y origin
 *    yinc       i      Y increment
 *    nx         i      NX (columns)
 *    ny         i      NY (rows)
 *    rot_deg    i      rotation
 *    x0, y0     i      point in space (z constant)
 *    azimut     i      azimuth (degrees) from North (positive clockwise)
 *    p_map_v   i/o     pointer to map values to update
 *    nn         i      map dimensions
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

#include "logger.h"
#include <math.h>
#include "libxtg.h"
#include "libxtg_.h"

int surf_get_dist_values(
			 double xori,
			 double xinc,
			 double yori,
			 double yinc,
			 int    nx,
			 int    ny,
			 double rot_deg,
			 double x0,
			 double y0,
			 double azimuth,
			 double *p_map_v,
                         long   nn,
			 int    flag
			 )
{
    /* locals */

    double   *xv, *yv, azi, angle, trueangle, dx, dy, distance;
    double   x1, y1, z1, x2, y2, z2, x3, y3, z3, pdist=0.1;
    int      i, j,ib, ier;

    /* azimuth rotation: */
    azi=(azimuth)*PI/180.0;  /* radians, positive */

    /* get the coordinates */
    xv = calloc(nn, sizeof(double));
    yv = calloc(nn, sizeof(double));

    nn = nx * ny;
    ier = surf_xy_as_values(xori, xinc, yori, yinc, nx, ny, rot_deg, xv,
                            nn, yv, nn, 0, 0);

    if (ier != 0) {
	logger_error(__LINE__, "Something went wrong in %s", __FUNCTION__);
        return ier;
    }

    x1 = x0;
    y1 = y0;
    z1 = 0.0;

    angle = azi + (PI/2.0);
    trueangle = (PI/2.0) - angle;

    /* make a line piece of pdist m */
    dx = pdist*cos(trueangle);
    dy = pdist*sin(trueangle);

    x2 = x0 + dx;
    y2 = y0 + dy;
    z2 = z1;


    for (i=1; i<=nx; i++) {
        for (j=1; j<=ny; j++) {

	    ib = x_ijk2ic(i,j,1,nx,ny,1,0);

	    x3 = xv[ib];
	    y3 = yv[ib];
	    z3 = 0;



	    ier = x_point_line_dist(x1, y1, z1, x2, y2, z2, x3, y3, z3,
				    &distance, 0, 1, 0);

	    if (ier != 0) return ier;

	    if (p_map_v[ib]<UNDEF_LIMIT) {
		p_map_v[ib] = distance;
	    }
	}
    }

    free(xv);
    free(yv);

   return EXIT_SUCCESS;
}
