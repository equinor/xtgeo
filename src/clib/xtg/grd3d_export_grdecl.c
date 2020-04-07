
/*
****************************************************************************************
*
* NAME:
*    grd3d_export_grdecl.c
*
* DESCRIPTION:
*    Export to Eclipse GRDECL format, either ASCII text or Ecl binary style
*
* ARGUMENTS:
*    nx, ny, nz     i     NCOL, NROW, NLAY
*    coordsv        i     COORD array w/ len
*    zcornsv        i     ZCORN array w/ len
*    actnumsv       i     ACTNUM array w/ len
*    filename       i     File name
*    mode           i     File mode, 1 ascii, 0  is binary
*
* RETURNS:
*    Void function
*
* LICENCE:
*    CF. XTGeo license
****************************************************************************************
*/

#include "libxtg.h"
#include "libxtg_.h"
#include "logger.h"

void
grd3d_export_grdecl(int nx,
                    int ny,
                    int nz,
                    double *coordsv,
                    long ncoordin,
                    double *zcornsv,
                    long nzcornin,
                    int *actnumsv,
                    long nactin,
                    char *filename,
                    int mode)

{
    int i, j, k, jj;
    long ic = 0, ib = 0;
    FILE *fc;
    int idum;
    long ncc, ncoord, nzcorn, nact;
    float *farr, fdum;
    double ddum;
    int itmp[4];

    logger_info(LI, FI, FU, "Entering %s", FU);

    /*
     *-------------------------------------------------------------------------
     * Open file
     *-------------------------------------------------------------------------
     */

    if (mode == 0) {
        logger_info(LI, FI, FU, "Opening binary GRDECL file...");
        fc = x_fopen(filename, "wb");
    } else {
        logger_info(LI, FI, FU, "Opening text GRDECL file...");
        fc = x_fopen(filename, "w");
    }

    /*
     *-------------------------------------------------------------------------
     * SPECGRID
     *-------------------------------------------------------------------------
     */

    itmp[0] = nx;
    itmp[1] = ny;
    itmp[2] = nz;
    itmp[3] = 1;

    if (mode == 0) {
        grd3d_write_eclrecord(fc, "SPECGRID", 1, itmp, &fdum, &ddum, 4);
    } else {
        grd3d_write_eclinput(fc, "SPECGRID", 1, itmp, &fdum, &ddum, 4, "  %5d", 10);
    }

    /*
     *-------------------------------------------------------------------------
     * COORD
     *-------------------------------------------------------------------------
     */
    ncoord = (nx + 1) * (ny + 1) * 6;
    farr = calloc(ncoord, sizeof(float));

    ib = 0;
    ncc = 0;
    for (j = 0; j <= ny; j++) {
        for (i = 0; i <= nx; i++) {

            for (jj = 0; jj < 6; jj++)
                farr[ncc++] = coordsv[ib + jj];

            ib = ib + 6;
        }
    }

    if (mode == 0) {
        grd3d_write_eclrecord(fc, "COORD", 2, &idum, farr, &ddum, ncoord);
    } else {
        grd3d_write_eclinput(fc, "COORD", 2, &idum, farr, &ddum, ncoord, "  %15.3f", 6);
    }
    free(farr);

    /*
     *-------------------------------------------------------------------------
     * ZCORN
     *-------------------------------------------------------------------------
     * ZCORN is ordered cycling X fastest, then Y, then Z, for all
     * 8 corners. XTGeo format is a bit different, having 4 corners
     * pr cell layer as       3_____4
     *                        |     |
     *                        |     |
     *                        |_____|
     *                        1     2
     */

    nzcorn = nx * ny * nz * 8; /* 8 Z values per cell for ZCORN */
    farr = calloc(nzcorn, sizeof(float));
    for (k = 1; k <= nz; k++) {
        /* top */
        for (j = 1; j <= ny; j++) {
            for (i = 1; i <= nx; i++) {
                ib = x_ijk2ib(i, j, k, nx, ny, nz + 1, 0);

                farr[ic++] = zcornsv[4 * ib + 1 * 1 - 1];
                farr[ic++] = zcornsv[4 * ib + 1 * 2 - 1];
            }

            for (i = 1; i <= nx; i++) {
                ib = x_ijk2ib(i, j, k, nx, ny, nz + 1, 0);

                farr[ic++] = zcornsv[4 * ib + 1 * 3 - 1];
                farr[ic++] = zcornsv[4 * ib + 1 * 4 - 1];
            }
        }

        /* bottom */
        for (j = 1; j <= ny; j++) {
            for (i = 1; i <= nx; i++) {
                ib = x_ijk2ib(i, j, k + 1, nx, ny, nz + 1, 0);

                farr[ic++] = zcornsv[4 * ib + 1 * 1 - 1];
                farr[ic++] = zcornsv[4 * ib + 1 * 2 - 1];
            }
            for (i = 1; i <= nx; i++) {
                ib = x_ijk2ib(i, j, k + 1, nx, ny, nz + 1, 0);

                farr[ic++] = zcornsv[4 * ib + 1 * 3 - 1];
                farr[ic++] = zcornsv[4 * ib + 1 * 4 - 1];
            }
        }
    }

    if (mode == 0) {
        grd3d_write_eclrecord(fc, "ZCORN", 2, &idum, farr, &ddum, nzcorn);
    } else {
        grd3d_write_eclinput(fc, "ZCORN", 2, &idum, farr, &ddum, nzcorn, "  %11.3f", 6);
    }
    free(farr);

    /*
     *-------------------------------------------------------------------------
     * ACTNUM
     *-------------------------------------------------------------------------
     */
    nact = nx * ny * nz;

    if (mode == 0) {
        grd3d_write_eclrecord(fc, "ACTNUM", 1, actnumsv, &fdum, &ddum, nact);
    } else {
        grd3d_write_eclinput(fc, "ACTNUM", 1, actnumsv, &fdum, &ddum, nact, "  %1d",
                             12);
    }

    /*
     *-------------------------------------------------------------------------
     * Close file
     *-------------------------------------------------------------------------
     */

    fclose(fc);
}
