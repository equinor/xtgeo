/*
****************************************************************************************
*
* NAME:
*    surf_resample.c
*
* DESCRIPTION:
*    Resample from one grid to another via bilinear interpolation
*
* ARGUMENTS:
*    nx1,ny1        i     Dimensions of first grid (origin)
*    xori1,xinc1    i     Maps X settings origin grid
*    yori1,yinc1    i     Maps Y settings origin grid
*    rot1           i     Rotation of origin
*    mapv1          i     Map array (origin)
*    nx2,ny2        i     Dimensions of second grid (result)
*    xori2,xinc2    i     Maps X settings result grid
*    yori2,yinc2    i     Maps Y settings result grid
*    rot2           i     Rotation of result
*    mapv2         i/o    Result grid
*    option         i     If 1 then outside nodes will be UNDEF (masked)
*
* RETURNS:
*    Int + Changed pointer to result map
*
* TODO:
*    YFLIP handling?
*
* LICENCE:
*    cf. XTGeo LICENSE
***************************************************************************************
*/

#include "libxtg.h"
#include "libxtg_.h"
#include "logger.h"

int
surf_resample(int nx1,
              int ny1,
              double xori1,
              double xinc1,
              double yori1,
              double yinc1,
              int yflip1,
              double rota1,
              double *mapv1,
              long nn1,
              int nx2,
              int ny2,
              double xori2,
              double xinc2,
              double yori2,
              double yinc2,
              int yflip2,
              double rota2,
              double *mapv2,
              long nn2,
              int optmask)

{
    /* locals */
    int i2, j2, ier2, ib2;
    double xc2, yc2, zc2, zc;

    logger_info(LI, FI, FU, "Resampling surface...");

    for (i2 = 1; i2 <= nx2; i2++) {
        for (j2 = 1; j2 <= ny2; j2++) {

            ib2 = x_ijk2ic(i2, j2, 1, nx2, ny2, 1, 0); /* C order */

            if (optmask == 1)
                mapv2[ib2] = UNDEF;

            /* get the x y location in the result: */
            ier2 = surf_xyz_from_ij(i2, j2, &xc2, &yc2, &zc2, xori2, xinc2, yori2,
                                    yinc2, nx2, ny2, yflip2, rota2, mapv2, nn2, 1);

            /* based on this X Y, need to find Z value from original: */
            if (ier2 == 0) {
                zc = surf_get_z_from_xy(xc2, yc2, nx1, ny1, xori1, yori1, xinc1, yinc1,
                                        yflip1, rota1, mapv1, nn1, 0);
                mapv2[ib2] = zc;

            } else {
                logger_info(LI, FI, FU, "Something went wrong: ier2 = %d", ier2);
                return ier2;
            }
        }
    }
    logger_info(LI, FI, FU, "Resampling surface... done!");
    return 0;
}
