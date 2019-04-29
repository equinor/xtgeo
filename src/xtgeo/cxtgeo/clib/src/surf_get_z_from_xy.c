/*
 ******************************************************************************
 *
 * NAME:
 *    surf_get_z_from_xy.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
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
 *    rot_deg       i      Rotation
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

double surf_get_z_from_xy(
			  double x,
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
			  int debug
			  )
{
    int  ib=-9, ier, iex[4], i=0, j=0;
    char sub[24]="surf_get_z_from_xy";
    double x_v[4], y_v[4], z_v[4];
    double xx, yy, zz, z, rx, ry;
    int userelative=1;

    xtgverbose(debug);


    if (debug > 1) {
        xtg_speak(sub,3,"Entering routine %s", sub);
    }
    if (nx*ny != nn) xtg_error(sub, "Fatal error");


    ib=-1;


    /* get i and j for lower left corner, given a point X Y*/

    ier = sucu_ij_from_xy(&i, &j, &rx, &ry,
			  x, y, xori, xinc, yori, yinc,
			  nx, ny, yflip, rot_deg, 1, debug);


    /* outside map, returning UNDEF value */
    if (ier<0) {
	return UNDEF;
    }

    /* two approaches here; the 'userelative' is more clean and safe? */

    if (userelative == 1) {
        /* map origin rleative is 0.0 */
        z = surf_get_z_from_ij(i, j, rx, ry, nx, ny, xinc, yinc, 0.0,
                               0.0, p_map_v, debug);
    }
    else{

        /* find the x,y,z values of the four nodes */
        iex[0] = surf_xyz_from_ij(i, j, &xx, &yy, &zz, xori, xinc, yori, yinc,
                                  nx, ny, yflip, rot_deg, p_map_v, nn, 0,
                                  debug);
        x_v[0]=xx; y_v[0]=yy; z_v[0]=zz;

        iex[1] = surf_xyz_from_ij(i+1, j, &xx, &yy, &zz, xori, xinc, yori,
                                  yinc, nx, ny, yflip, rot_deg, p_map_v,
                                  nn, 0, debug);
        x_v[1]=xx; y_v[1]=yy; z_v[1]=zz;

        iex[2] = surf_xyz_from_ij(i, j+1, &xx, &yy, &zz, xori, xinc, yori,
                                  yinc, nx, ny, yflip, rot_deg, p_map_v,
                                  nn, 0, debug);
        x_v[2]=xx; y_v[2]=yy; z_v[2]=zz;

        iex[3] = surf_xyz_from_ij(i+1, j+1, &xx, &yy, &zz, xori, xinc, yori,
                                  yinc, nx, ny, yflip, rot_deg, p_map_v,
                                  nn, 0, debug);
        x_v[3]=xx; y_v[3]=yy; z_v[3]=zz;

        for (i=0; i<4; i++) {
            if (iex[i] != 0) {
                return UNDEF;
            }
        }

        // now find the Z value, using interpolation method 3 (bilinear w/rot.)

        z = x_interp_map_nodes(x_v, y_v, z_v, x, y, 3, debug);
    }


    return z;
}
