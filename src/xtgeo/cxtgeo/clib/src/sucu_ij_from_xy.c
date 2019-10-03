/*
****************************************************************************************
 *
 * Find IJ coordinate in surface or cube given XY
 *
 ***************************************************************************************
 */

#include <math.h>
#include "libxtg.h"
#include "libxtg_.h"

/*
****************************************************************************************
 *
 * NAME:
 *    sucu_ij_from_xy.c
 *
 * DESCRIPTION:
 *     Routine(s) computing IJ in a rotated surf or cube given X,Y.
 *     The points will be the cell or the corner node, depending on how you
 *     think...
 *     Will return -1 if XY is outside surf or cube, 0 if success.
 *
 *     By _outside_, the point is outside the XY nodes in the cube. One
 *     could of course think that a cube extends 0.5 cell-size when the
 *     nodes are regarded as cell centers, but this is not taken into accont
 *     her (yet). A numerical presision is however taken into account along
 *     the border.
 *
 * ARGUMENTS:
 *    i, j           o     col/row to update
 *    rx, ry         o     relative coords of input point (useful in interpol)
 *    x, y           i     Input point
 *    xori           i     X origin coordinate
 *    xinc           i     X increment
 *    yori           i     Y origin coordinate
 *    yinc           i     Y increment
 *    nx, ny         i     Dimensions
 *    yflip          i     Determining YFLIP (1 or -1)
 *    rot            i     Rotation (degrees, from X axis, anti-clock)
 *    flag           i     Options flag:
 *                         0: get cell nearest cell center coordinate
 *                         1: get lowerleft (i,j) corner, means that point
 *                            is within i,j  i,j+1  i+1,j  i+1,j+1
 *
 * RETURNS:
 *    Function: 0: upon success. If problems or outside <> 0:. Update pointers
 *
 * TODO/ISSUES/BUGS:
 *    yflip handling?
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ***************************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"

int sucu_ij_from_xy(
                    int *i,
                    int *j,
                    double *rx,
                    double *ry,
                    double xin,
                    double yin,
                    double xori,
                    double xinc,
                    double yori,
                    double yinc,
                    int nx,
                    int ny,
                    int yflip,
                    double rot_deg,
                    int flag                    )
{
    /* locals */
    int ierx = 0, iery = 0;
    double angle, x1, x2, x3, y1, y2, y3, px, py, relx, rely, zin = 0.0;
    double x0, y0, z0;
    int ipos, jpos, option2;


    angle = rot_deg * PI / 180.0;

    yinc = yinc * yflip;

    /* find the RELATIVE corners of the map or cube */

    /*
     *   3               4
     *   |---------------|
     *   |               |
     *   |         P     |
     *   |               |
     *   |---------------|
     *   1               2
     */

    /* first: actual corners (do not need corner 4): */

    x1 = 0.0;
    y1 = 0.0;

    x2 = xinc * (nx - 1) * cos(angle);
    y2 = xinc * (nx - 1) * sin(angle);

    x3 = yinc * (ny - 1) * cos(angle + PI / 2.0);
    y3 = yinc * (ny - 1) * sin(angle + PI / 2.0);

    /* xin = xin - xori + 0.5 * FLOATEPS;  // Workaround: To avoid numerical trouble */
    /* y = y - yori + 0.5 * FLOATEPS;  // when points are exact overlapping... */
    xin = xin - xori;
    yin = yin - yori;

    /* determine _relative_ coordinate of point x y on X axis: */
    option2 = 2;  /* allow for numerical precision error for border */

    int debug = 0;
    ierx = x_point_line_pos(x1, y1, 0.0, x2, y2, 0.0, xin, yin, zin,
                            &x0, &y0, &z0, &relx, option2, debug);

    /* determine _relative_ coordinate of point x y on Y axis: */
    iery = x_point_line_pos(x1, y1, 0.0, x3, y3, 0.0, xin, yin, zin,
                            &x0, &y0, &z0, &rely, option2, debug);

    if (ierx == -1 || iery == -1) {
        return -1;
    }
    /* now use the new coordinates */

    px = relx * xinc * (nx - 1);
    py = rely * yinc * (ny - 1);

    if (flag == 0) {
        /* ~find cell index */
        ipos = (int)((px + 0.5 * xinc) / xinc) + 1;
        jpos = (int)((py + 0.5 * yinc) / yinc) + 1;
        *i = ipos;
        *j = jpos;
    }
    else{
        /* ~find nodes; which is the lower left index position: */
        ipos = (int)((px) / xinc) + 1;
        jpos = (int)((py) / yinc) + 1;

        /* if (ipos >= nx) ipos = nx - 1; */
        /* if (jpos >= nx) jpos = ny - 1; */

        /* if (ipos < 1) ipos = 1; */
        /* if (jpos < 1) jpos = 1; */

        *i = ipos;
        *j = jpos;
    }

    *rx = px;
    *ry = py;

    return EXIT_SUCCESS;
}
