/*
****************************************************************************************
*
* NAME:
*    grd3d_get_all_corners.c
*
* DESCRIPTION:
*    Get all cell corners as single arrays
*
* ARGUMENTS:
*    nx, ny, nz     i     Dimensions
*    coordsv        i     Coordinate vector (xtgeo fmt)
*    zcornsv        i     ZCORN vector (xtgeo fmt)
*    actnumsv       i     ACTNUM vector (xtgeo fmt)
*    x1, ... z8     o     Single vectors for all X Y Z for all 8 corners
*    option         i     if 0, cells with ACTNUM 0 becomes UNDEF
*
* RETURNS:
*    Status, EXIT_FAILURE or EXIT_SUCCESS
*
* TODO/ISSUES/BUGS:
*
* LICENCE:
*    CF XTGeo's LICENSE
***************************************************************************************
*/

#include "libxtg.h"
#include "libxtg_.h"

void
grd3d_get_all_corners(int nx,
                      int ny,
                      int nz,

                      double *coordsv,
                      long ncoordin,
                      double *zcornsv,
                      long nzcornin,
                      int *actnumsv,
                      long nactin,

                      double *x1,
                      double *y1,
                      double *z1,
                      double *x2,
                      double *y2,
                      double *z2,
                      double *x3,
                      double *y3,
                      double *z3,
                      double *x4,
                      double *y4,
                      double *z4,
                      double *x5,
                      double *y5,
                      double *z5,
                      double *x6,
                      double *y6,
                      double *z6,
                      double *x7,
                      double *y7,
                      double *z7,
                      double *x8,
                      double *y8,
                      double *z8,
                      int option)

{
    double crs[24];
    int i, j, k;

    for (k = 1; k <= nz; k++) {
        for (j = 1; j <= ny; j++) {
            for (i = 1; i <= nx; i++) {

                long ib = x_ijk2ib(i, j, k, nx, ny, nz, 0);
                if (ib < 0) {
                    throw_exception("Loop resulted in index outside "
                                    "boundary in grd3d_get_all_corners");
                    return;
                }

                if (option == 1 && actnumsv[ib] == 0) {
                    x1[ib] = UNDEF;
                    y1[ib] = UNDEF;
                    z1[ib] = UNDEF;
                    x2[ib] = UNDEF;
                    y2[ib] = UNDEF;
                    z2[ib] = UNDEF;
                    x3[ib] = UNDEF;
                    y3[ib] = UNDEF;
                    z3[ib] = UNDEF;
                    x4[ib] = UNDEF;
                    y4[ib] = UNDEF;
                    z4[ib] = UNDEF;
                    x5[ib] = UNDEF;
                    y5[ib] = UNDEF;
                    z5[ib] = UNDEF;
                    x6[ib] = UNDEF;
                    y6[ib] = UNDEF;
                    z6[ib] = UNDEF;
                    x7[ib] = UNDEF;
                    y7[ib] = UNDEF;
                    z7[ib] = UNDEF;
                    x8[ib] = UNDEF;
                    y8[ib] = UNDEF;
                    z8[ib] = UNDEF;
                } else {
                    grd3d_corners(i, j, k, nx, ny, nz, coordsv, 0, zcornsv, 0, crs);

                    x1[ib] = crs[0];
                    y1[ib] = crs[1];
                    z1[ib] = crs[2];
                    x2[ib] = crs[3];
                    y2[ib] = crs[4];
                    z2[ib] = crs[5];
                    x3[ib] = crs[6];
                    y3[ib] = crs[7];
                    z3[ib] = crs[8];
                    x4[ib] = crs[9];
                    y4[ib] = crs[10];
                    z4[ib] = crs[11];
                    x5[ib] = crs[12];
                    y5[ib] = crs[13];
                    z5[ib] = crs[14];
                    x6[ib] = crs[15];
                    y6[ib] = crs[16];
                    z6[ib] = crs[17];
                    x7[ib] = crs[18];
                    y7[ib] = crs[19];
                    z7[ib] = crs[20];
                    x8[ib] = crs[21];
                    y8[ib] = crs[22];
                    z8[ib] = crs[23];
                }
            }
        }
    }
}
