/*
****************************************************************************************
*
* NAME:
*    grdcp3d_from_cube.c
*
* DESCRIPTION:
*    Create a simple shoebox grid, based on DX, DY rotation etc. I.e.
*    similar to a Cube spec. Using xtgformat=2.
*
* ARGUMENTS:
*    ncol..nlay     i     Dimensions
*    coordsv       i/o    Coordinates (must be allocated in caller)
*    zcornsv       i/o    Z corners (must be allocated in called)
*    actnumsv      i/o    ACTNUM (must be allocated in caller)
*    xori..zori     i     Origins
*    xinc..zinc     i     Increments
*    option         i     0: input is lower left (top) corner node; 1 input
*                         lower left (top) cell center
*    debug          i     Debug level
*
* RETURNS:
*    Void. Pointers to arrays are updated
*
* TODO/ISSUES/BUGS:
*
* LICENCE:
*    cf. XTGeo LICENSE
***************************************************************************************
*/
#include <xtgeo/xtgeo.h>
#include "common.h"
#include "logger.h"

void
grdcp3d_from_cube(int ncol,
                  int nrow,
                  int nlay,
                  double *coordsv,
                  long ncoord,
                  float *zcornsv,
                  long nzcorn,
                  int *actnumsv,
                  long nactnum,
                  double xori,
                  double yori,
                  double zori,
                  double xinc,
                  double yinc,
                  double zinc,
                  double rotation,
                  int yflip,
                  int option)

{
    /* locals */
    int nn, iok, i, j, k;
    double xcoord = 0.0, ycoord = 0.0;

    logger_info(LI, FI, FU, "Making Grid3D from cube or shoebox spec");

    if (option == 1) { /* input is cell center, not cell corner */
        double res[8];
        x_2d_rect_corners(xori, yori, xinc, yinc, rotation, res);
        xori = res[6];
        yori = res[7];
        if (yflip == -1) {
            xori = res[0];
            yori = res[1];
        }
        zori = zori - 0.5 * zinc;
    }

    /* make coord... */

    long ibc = 0;
    for (i = 1; i <= ncol + 1; i++) {
        for (j = 1; j <= nrow + 1; j++) {

            iok = cube_xy_from_ij(i, j, &xcoord, &ycoord, xori, xinc, yori, yinc,
                                  ncol + 1, nrow + 1, yflip, rotation, 0);

            if (iok != 0) {
                throw_exception("Bug in: grdcp3d_from_cube");
                return;
            }
            coordsv[ibc++] = xcoord;
            coordsv[ibc++] = ycoord;
            coordsv[ibc++] = zori;

            coordsv[ibc++] = xcoord;
            coordsv[ibc++] = ycoord;
            coordsv[ibc++] = zori + zinc * (nlay + 1);
        }
    }

    /* make zcorn and actnum... */
    long ibz = 0;
    long iba = 0;
    for (i = 1; i <= ncol + 1; i++) {
        for (j = 1; j <= nrow + 1; j++) {
            double zlevel = zori;
            for (k = 1; k <= nlay + 1; k++) {

                for (nn = 0; nn < 4; nn++)
                    zcornsv[ibz++] = zlevel;

                if (i <= ncol && j <= nrow && k <= nlay)
                    actnumsv[iba++] = 1;

                zlevel += zinc;
            }
        }
    }
}
