/*
 ***************************************************************************************
 *
 * NAME:
 *    grd3d_midpoint.c
 *
 * DESCRIPTION:
 *    Fint the midpoint is a spesific cell (cell count IJK with base 1)
 *
 * ARGUMENTS:
 *    i, j, k        i     Cell IJK
 *    nx, ny, nz     i     Dimensions
 *    coordsv        i     Coordinate vector (with numpy dimensions)
 *    zcornsv        i     ZCORN vector (with numpy dimensions)
 *    x, y, z        o     XYZ output
 *
 * RETURNS:
 *    Void
 *
 * TODO/ISSUES/BUGS:
 *    Is simple midpoint the best approach? Consider alternatives
 * LICENCE:
 *    CF XTGeo's LICENSE
 ***************************************************************************************
 */
#include <xtgeo/xtgeo.h>

#include "logger.h"

void
grd3d_midpoint(int i,
               int j,
               int k,
               int nx,
               int ny,
               int nz,
               double *coordsv,
               long ncoord,
               double *zcornsv,
               long nzcorn,
               double *x,
               double *y,
               double *z)

{
    double c[24];

    /* get all 24 corners */

    grd3d_corners(i, j, k, nx, ny, nz, coordsv, ncoord, zcornsv, nzcorn, c);

    /* compute the midpoint for X,Y,Z (is this OK algorithm?)*/

    *x = 0.125 * (c[0] + c[3] + c[6] + c[9] + c[12] + c[15] + c[18] + c[21]);
    *y = 0.125 * (c[1] + c[4] + c[7] + c[10] + c[13] + c[16] + c[19] + c[22]);
    *z = 0.125 * (c[2] + c[5] + c[8] + c[11] + c[14] + c[17] + c[20] + c[23]);
}
