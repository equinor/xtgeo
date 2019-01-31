/*
 ******************************************************************************
 *
 * Set a map value inside a polygon
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
 *    surf_setval_poly.c
 *
 * DESCRIPTION:
 *     Given a closed polygon, set a map value inside that polygon. This may
 *     be used to "flag" inside polygon, e.g. for numpy operations in Python.
 *
 * ARGUMENTS:
 *    xori           i     X origin coordinate
 *    xinc           i     X increment
 *    yori           i     Y origin coordinate
 *    yinc           i     Y increment
 *    ncol, nrow     i     Dimensions
 *    yflip          i     YFLIP indicator 1 or -1
 *    rot            i     Rotation (degrees, from X axis, anti-clock)
 *    p_map_v       i/o    Surf map array
 *    nmap           i     Lenghth of map array (ncol * nrow; for swig)
 *    p_xp_v         i     Polygons array of X
 *    nnx            i     Polygons X array length
 *    p_yp_v         i     Polygons array of Y
 *    nny            i     Polygons X array length (nny = nnx; entry for swig)
 *    value          i     Value to set inside polygon
 *    option         i     Options flag; for future usage
 *    debug          i     Debug level
 *
 * RETURNS:
 *    Function: 0: upon success. If problems <> 0:
 *    -5: Error in findig xyz from I J coordinate (something is wrong)
 *    -9: Polygon is not closed.
 *
 *    If success, p_map_v array is updated inside polygon.
 *
 * LICENCE:
 *    CF XTGeo licence
 ******************************************************************************
 */


#include "libxtg.h"
#include "libxtg_.h"

int surf_setval_poly(
                     double xori,
                     double xinc,
                     double yori,
                     double yinc,
                     int ncol,
                     int nrow,
                     int yflip,
                     double rot_deg,
                     double *p_map_v,
                     long nmap,
                     double *p_xp_v,
                     long npolx,
                     double *p_yp_v,
                     long npoly,
                     double value,
                     int flag,
                     int debug
                     )
{
    /* locals */
    int ino, jno;
    long ic;
    double xcor, ycor, zval;
    int ier, status;

    char sbn[24] = "surf_setval_poly";
    xtgverbose(debug);

    if (debug > 2) xtg_speak(sbn, 3, "Entering routine %s", sbn);

    for (ino = 1; ino <= ncol; ino++) {
        for (jno = 1; jno <= nrow; jno++) {

            ic = x_ijk2ic(ino, jno, 1, ncol, nrow, 1, 0);

            ier = surf_xyz_from_ij(ino, jno, &xcor, &ycor, &zval,
                                   xori, xinc, yori, yinc,
                                   ncol, nrow, yflip, rot_deg,
                                   p_map_v, nmap, 0, debug);

            if (ier != 0) return -5;

            status = pol_chk_point_inside(xcor, ycor, p_xp_v, p_yp_v, npolx,
                                          debug);

            if (status == -9) return -9;  /* polygon is not closed */

            if (status > 0 && p_map_v[ic] < UNDEF_LIMIT) p_map_v[ic] = value;
        }
    }

    return EXIT_SUCCESS;
}
