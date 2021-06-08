/*
 ******************************************************************************
 *
 * NAME:
 *    surf_get_z_from_ij.c (same as map_get_z_from_ij, but C order lookup)
 *
 *
 * DESCRIPTION:
 *    Given lower left IJ point and P(x,y), the map Z value is returned.
 *    This should work for rotated maps and cubes if the RELATIVE coords
 *    are provided.
 *
 *    Since relative coords are required, rotation is not relevant.
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
 *    option        i      0: sampling by bilinear interpolation,
 *                         1: sampling by nearest node
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

double
surf_get_z_from_ij(int ic,
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
                   int option)
{

    double x_v[4], y_v[4], z_v[4];
    double z;

    // find the values of four nodes
    x_v[0] = xori + (ic - 1) * xinc;
    x_v[1] = xori + (ic)*xinc;
    x_v[2] = xori + (ic - 1) * xinc;
    x_v[3] = xori + (ic)*xinc;
    y_v[0] = yori + (jc - 1) * yinc;
    y_v[1] = yori + (jc - 1) * yinc;
    y_v[2] = yori + (jc)*yinc;
    y_v[3] = yori + (jc)*yinc;

    int iba = x_ijk2ic(ic, jc, 1, nx, ny, 1, 0);
    if (iba < 0) {
        return UNDEF;
    }
    z_v[0] = p_map_v[iba];

    int ibb = x_ijk2ic(ic + 1, jc, 1, nx, ny, 1, 0);
    if (ibb < 0) {
        z_v[1] = p_map_v[iba];
    } else {
        z_v[1] = p_map_v[ibb];
    }

    int ibc = x_ijk2ic(ic, jc + 1, 1, nx, ny, 1, 0);
    if (ibc < 0) {
        z_v[2] = p_map_v[iba];
    } else {
        z_v[2] = p_map_v[ibc];
    }

    int ibd = x_ijk2ic(ic + 1, jc + 1, 1, nx, ny, 1, 0);
    if (ibd < 0) {
        z_v[3] = p_map_v[iba];
    } else {
        z_v[3] = p_map_v[ibd];
    }

    // now find the Z value, using interpolation method 2 (bilinear) which assumes
    // no rotation, or opt_interp 4 for nearest sampling

    int opt_interp = 2;
    if (option == 1)
        opt_interp = 4;

    z = x_interp_map_nodes(x_v, y_v, z_v, x, y, opt_interp);

    return z;
}
