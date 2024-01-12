/*
 ***************************************************************************************
 *
 * NAME:
 *    grdcp3d_midpoint.c
 *
 * DESCRIPTION:
 *    Find the midpoint is a specific cell (cell count IJK with base 1)
 *
 * ARGUMENTS:
 *    i, j, k           i     Cell IJK
 *    ncol, nrow, nlay  i     Dimensions
 *    coordsv           i     Coordinate vector (with numpy dimensions)
 *    zcornsv           i     ZCORN vector (with numpy dimensions)
 *    x, y, z           o     XYZ output
 *
 * RETURNS:
 *    Void
 *
 * TODO/ISSUES/BUGS:
 *    None
 *
 * LICENCE:
 *    CF XTGeo's LICENSE
 ***************************************************************************************
 */
#include <xtgeo/xtgeo.h>

void
grdcp3d_midpoint(long i,
                 long j,
                 long k,
                 long ncol,
                 long nrow,
                 long nlay,
                 double *coordsv,
                 long ncoord,
                 float *zcornsv,
                 long nzcorn,
                 double *x,
                 double *y,
                 double *z)

{
    double c[24];

    /* Get the (x, y, z) coordinates of all 8 corners in `c` */
    grdcp3d_corners(i, j, k, ncol, nrow, nlay, coordsv, ncoord, zcornsv, nzcorn, c);

    /* Find centroid via arithmetic mean (average of each dimensional coord) */
    *x = 0.125 * (c[0] + c[3] + c[6] + c[9] + c[12] + c[15] + c[18] + c[21]);
    *y = 0.125 * (c[1] + c[4] + c[7] + c[10] + c[13] + c[16] + c[19] + c[22]);
    *z = 0.125 * (c[2] + c[5] + c[8] + c[11] + c[14] + c[17] + c[20] + c[23]);
}
