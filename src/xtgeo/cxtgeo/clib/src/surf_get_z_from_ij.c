/*
 ******************************************************************************
 *
 * NAME:
 *    surf_get_z_from_ij.c (same as map_get_z_from_ij, but C order lookup)
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Given lower left IJ point and P(x,y), the map Z value is returned.
 *    This should work for rotated maps and cubes if the RELATIVE coords
 *    are provided
 *
 *   |-------|
 *   | *P    |
 * ->|_______|
 *   ^
 *   |
 *
 * The points should be organized as follows (nonrotated maps):
 *
 *     2       3          N
 *                        |
 *     0       1          |___E
 *
 * ARGUMENTS:
 *    ic, jc        i      Lower left corner (0)
 *    x, y          i      Actual point i cell (real or relative)
 *    nx, ny        i      Dimensions
 *    xori, yori    i      Map origins (real or relative)
 *    xinc, yinc    i      Map increments
 *    p_map_v       i      Pointer to map values to update
 *    flag          i      Flag for options
 *    debug         i      Debug flag
 *
 * RETURNS:
 *    Z value at point
 *
 * TODO/ISSUES/BUGS:
 *    - checking the handling of undef nodes; shall return UNDEF
 *    - Propert handling of YFLIP = -1!
 * LICENCE:
 *    cf. XTGeo LICENSE
 ******************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"

double surf_get_z_from_ij(
			int ic,
			int jc,
                        double x,
                        double y,
			int nx,
			int ny,
			double xinc,
			double yinc,
			double xori,
			double yori,
			double *p_map_v,

			int debug
			)
{


    int  ibc=-9;
    char s[24]="map_get_z_from_ij";
    double x_v[4], y_v[4], z_v[4];
    double z;

    xtgverbose(debug);
    if (debug > 2) xtg_speak(s, 3, "Entering routine %s", s);

    // find the values of four nodes

    x_v[0] = xori + (ic-1) * xinc;
    x_v[1] = xori + (ic) * xinc;
    x_v[2] = xori + (ic-1) * xinc;
    x_v[3] = xori + (ic) * xinc;

    y_v[0] = yori + (jc-1) * yinc;
    y_v[1] = yori + (jc-1) * yinc;
    y_v[2] = yori + (jc) * yinc;
    y_v[3] = yori + (jc) * yinc;

    ibc = x_ijk2ic(ic, jc, 1, nx, ny, 1, 0);
    z_v[0] = p_map_v[ibc];

    ibc = x_ijk2ic(ic + 1, jc, 1, nx, ny, 1, 0);
    z_v[1] = p_map_v[ibc];

    ibc = x_ijk2ic(ic, jc + 1, 1, nx, ny, 1, 0);
    z_v[2] = p_map_v[ibc];

    ibc = x_ijk2ic(ic + 1, jc + 1, 1, nx, ny, 1, 0);
    z_v[3] = p_map_v[ibc];

    // now find the Z value, using interpolation method 2 (bilinear)

    z = x_interp_map_nodes(x_v, y_v, z_v, x, y, 2, debug);

    return z;

}
