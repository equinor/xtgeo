/*
****************************************************************************************
*
* NAME:
*    surf_get_zv_from_xyv.c
*
* DESCRIPTION:
*    Vector version of surf_get_z_from_xy.c
*
* ARGUMENTS:
*    xv, yv        i      XY Coordinates as arrays (vectors)
*    n*            i      Length of arrays
*    zv            o      Result vector
*    n*            i      Length of arrays
*    nx, ny        i      Surf dimensions
*    xori, yori    i      Map origins
*    xinc, yinc    i      Map increments
*    yflip         i      YFLIP 1 or -1
*    rot_deg       i      Rotation
*    p_map_v       i      Pointer to map values to update
*    option        i      0: interpolation using relative coordinates (rotation is ok!)
*                         1: interpolation using rotated map directly
*                         2: Nearest node sampling
*
*
* RETURNS:
*    Z value at point
*
* TODO/ISSUES/BUGS:
*    - checking the handling of undef nodes; shall return UNDEF
* LICENCE:
*    cf. XTGeo LICENSE
****************************************************************************************
*/
#include <xtgeo/xtgeo.h>

int
surf_get_zv_from_xyv(double *xv,
                     long nxv,
                     double *yv,
                     long nyv,
                     double *zv,
                     long nzv,
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
    int i;

    nn = nx * ny;

    for (i = 0; i < nxv; i++) {
        zv[i] = surf_get_z_from_xy(xv[i], yv[i], nx, ny, xori, yori, xinc, yinc, yflip,
                                   rot_deg, p_map_v, nn, option);
    }

    return (0);
}
