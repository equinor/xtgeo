/*
 ******************************************************************************
 *
 * NAME:
 *    surf_get_z_from_xy.c
 *
 *
 * DESCRIPTION:
 *    Given a map and a x,y point, the map Z value is returned. This should
 *    work for rotated maps... (cf map_get_z_from_xy.c that do not work w rot)
 *
 *   |-------|
 *   | *     |
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
 *    x, y          i      Coordinates
 *    nx, ny        i      Dimensions
 *    xori, yori    i      Map origins
 *    xinc, yinc    i      Map increments
 *    yflip         i      1 or -1 for map Y axis flip
 *    rot_deg       i      Rotation
 *    p_map_v       i      Pointer to map values to update
 *    option        i      0: interpolation using relative coordinates (rotation is ok!)
 *                         1: interpolation using rotated map directly
 *                         2: Nearest node sampling
 *
 * RETURNS:
 *    Z value at point
 *
 * TODO/ISSUES/BUGS:

 * LICENCE:
 *    LGPLv3
 ******************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"
#include "logger.h"

double
surf_get_z_from_xy(double x,
                   double y,
                   int nx,
                   int ny,
                   double xori,
                   double yori,
                   double xinc,
                   double yinc,
                   int yflip,
                   double rot_deg,
                   double *p_map_v,
                   long nn,
                   int option)
{
    int ier, iex[4], i = 0, j = 0;
    double x_v[4], y_v[4], z_v[4];
    double xx, yy, zz, z, rx, ry;

    if (nx * ny != nn)
        logger_error(LI, FI, FU, "Fatal error in %s", FU);

    /* get i and j for lower left corner, given a point X Y*/

    ier = sucu_ij_from_xy(&i, &j, &rx, &ry, x, y, xori, xinc, yori, yinc, nx, ny, yflip,
                          rot_deg, 1);

    /* outside map, returning UNDEF value */
    if (ier < 0) {
        return UNDEF;
    }

    /* two approaches here; the 'userelative' option = 1 is more clean and safe? */

    if (option == 0) {
        /* map origin relative is 0.0 */
        z = surf_get_z_from_ij(i, j, rx, ry, nx, ny, xinc, yinc * yflip, 0.0, 0.0,
                               p_map_v, 0);

    } else if (option == 1) {
        /* this is kept for legacy reference */

        /* find the x,y,z values of the four nodes */
        iex[0] = surf_xyz_from_ij(i, j, &xx, &yy, &zz, xori, xinc, yori, yinc, nx, ny,
                                  yflip, rot_deg, p_map_v, nn, 0);
        x_v[0] = xx;
        y_v[0] = yy;
        z_v[0] = zz;

        iex[1] = surf_xyz_from_ij(i + 1, j, &xx, &yy, &zz, xori, xinc, yori, yinc, nx,
                                  ny, yflip, rot_deg, p_map_v, nn, 0);
        x_v[1] = xx;
        y_v[1] = yy;
        z_v[1] = zz;

        iex[2] = surf_xyz_from_ij(i, j + 1, &xx, &yy, &zz, xori, xinc, yori, yinc, nx,
                                  ny, yflip, rot_deg, p_map_v, nn, 0);
        x_v[2] = xx;
        y_v[2] = yy;
        z_v[2] = zz;

        iex[3] = surf_xyz_from_ij(i + 1, j + 1, &xx, &yy, &zz, xori, xinc, yori, yinc,
                                  nx, ny, yflip, rot_deg, p_map_v, nn, 0);
        x_v[3] = xx;
        y_v[3] = yy;
        z_v[3] = zz;

        for (i = 0; i < 4; i++) {
            if (iex[i] != 0) {
                return UNDEF;
            }
        }

        // now find the Z value, using interpolation method 3 (bilinear w/rot.)

        z = x_interp_map_nodes(x_v, y_v, z_v, x, y, 3);

    } else if (option == 2) {
        /* map in relative coordinates but use nearest sampling*/
        z = surf_get_z_from_ij(i, j, rx, ry, nx, ny, xinc, yinc * yflip, 0.0, 0.0,
                               p_map_v, 1);
    }

    return z;
}
