/*
 ***************************************************************************************
 *
 * NAME:
 *    grd3d_translate.c
 *
 *
 * DESCRIPTION:
 *    Translate the coordinates in 3D (linear)
 *
 * ARGUMENTS:
 *    nx, ny, nx     i     Grid I J K
 *    *flip          i     Flip for X Y Z coords
 *    *shift         i     Shift for X Y Z coords
 *    coordsv       i/o    Coordinates vector
 *    zcornsv       i/o    Corners vector
 *    debug          i     Debug level
 *
 * RETURNS:
 *    Function: 0: upon success. If problems:
 *             -1: One of the flip are not 1 or -1
 *    Geometry vectors are updated
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 *
 * CODING STANDARD:
 *    cf. .clang-format file
 ***************************************************************************************
 */
#include <xtgeo/xtgeo.h>
#include "logger.h"

int
grd3d_translate(int nx,
                int ny,
                int nz,
                int xflip,
                int yflip,
                int zflip,
                double xshift,
                double yshift,
                double zshift,
                double *coordsv,
                long ncooordin,
                double *zcornsv,
                long nzcornin)

{
    /* locals */
    int i, j, ic, ib, iok = 0;

    logger_info(LI, FI, FU, "Do translation or pure flipping");

    if ((xflip == 1 || xflip == -1) && (yflip == 1 || yflip == -1) &&
        (zflip == 1 || zflip == -1)) {
        iok = 1;
    }

    if (iok == 0) {
        /* flip out of range */
        logger_warn(LI, FI, FU, "Error in flips ...%d %d %d", xflip, yflip, zflip);
        return (-1);
    }

    /* coord section */
    ib = 0;
    for (j = 0; j <= ny; j++) {
        for (i = 0; i <= nx; i++) {
            coordsv[ib + 0] = xflip * (coordsv[ib + 0] + xshift);
            coordsv[ib + 1] = yflip * (coordsv[ib + 1] + yshift);
            coordsv[ib + 2] = zflip * (coordsv[ib + 2] + zshift);
            coordsv[ib + 3] = xflip * (coordsv[ib + 3] + xshift);
            coordsv[ib + 4] = yflip * (coordsv[ib + 4] + yshift);
            coordsv[ib + 5] = zflip * (coordsv[ib + 5] + zshift);
            ib = ib + 6;
        }
    }

    /* zcorn section     */
    for (ic = 0; ic < nzcornin; ic++) {
        zcornsv[ic] = zflip * (zcornsv[ic] + zshift);
    }

    logger_info(LI, FI, FU, "Exit from routine");
    return (0);
}
