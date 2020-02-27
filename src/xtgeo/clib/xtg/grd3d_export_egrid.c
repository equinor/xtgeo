/*
****************************************************************************************
 *
 * NAME:
 *    grd3d_export_egrid.c
 *
 * DESCRIPTION:
 *    Export to Eclipse EGRID format, rather similar to binary GRDECL
 *
 * ARGUMENTS:
 *    nx, ny, nz     i     NCOL, NROW, NLAY
 *    coordsv      i     COORD array
 *    zcornsv      i     ZCORN array
 *    p_actnum_v     i     ACTNUM array
 *    filename       i     File name
 *    mode           i     File mode, 1 ascii, 0  is binary
 *
 * RETURNS:
 *    Void function
 *
 * LICENCE:
 *    CF. XTGeo license
 ***************************************************************************************
 */


#include "logger.h"
#include "libxtg.h"
#include "libxtg_.h"


void grd3d_export_egrid (
                         int nx,
                         int ny,
                         int nz,
                         double *coordsv,
                         long ncoordin,
                         double *zcornsv,
                         long nzcornin,
                         int *p_actnum_v,
                         long nactin,
                         char *filename,
                         int mode
                         )

{
    int i, j, k, jj;
    long ic = 0, ib = 0;
    FILE *fc;
    int idum;
    long ncc, ncoord, nzcorn, nact;
    float *farr, fdum;
    double ddum;
    int itmp[100];

    logger_info(LI, FI, FU, "Export to EGRID format, file: %s ...", filename);

    /*
     *-------------------------------------------------------------------------
     * Open file
     *-------------------------------------------------------------------------
     */

    fc = fopen(filename, "wb");

    if (fc == NULL) logger_critical(LI, FI, FU, "Cannot open file %s", filename);

    /*
     *-------------------------------------------------------------------------
     * FILEHEAD
     *-------------------------------------------------------------------------
     */

    for (i = 0; i < 100; i++) itmp[i] = 0;

    itmp[0] = 3; itmp[1] = 2017;

    grd3d_write_eclrecord(fc, "FILEHEAD", 1, itmp, &fdum,
                          &ddum, 100, XTGDEBUG);

    /*
     *-------------------------------------------------------------------------
     * GRIDHEAD
     *-------------------------------------------------------------------------
     */

    for (i = 0; i < 100; i++) itmp[i] = 0;

    itmp[0] = 1; itmp[1] = nx; itmp[2] = ny; itmp[3] = nz;
    // itmp[24] = 1; itmp[25] = 1;

    grd3d_write_eclrecord(fc, "GRIDHEAD", 1, itmp, &fdum,
                          &ddum, 100, XTGDEBUG);

    /*
     *-------------------------------------------------------------------------
     * COORD
     *-------------------------------------------------------------------------
     */
    ncoord = (nx + 1) * (ny + 1) * 6;
    farr = calloc(ncoord, sizeof(float));

    ib=0;
    ncc = 0;
    for (j = 0; j <= ny; j++) {
	for (i = 0;i <= nx; i++) {

            for (jj = 0; jj < 6; jj++) farr[ncc++] = coordsv[ib + jj];
            ib = ib + 6;

        }
    }

    grd3d_write_eclrecord(fc, "COORD", 2, &idum, farr, &ddum, ncoord,
                          XTGDEBUG);

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

    nzcorn = nx * ny * nz * 8;  /* 8 Z values per cell for ZCORN */
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
	    for (i=1; i<=nx; i++) {
		ib=x_ijk2ib(i, j, k+1, nx, ny, nz+1, 0);

                farr[ic++] = zcornsv[4 * ib + 1 * 1 - 1];
                farr[ic++] = zcornsv[4 * ib + 1 * 2 - 1];
	    }
	    for (i=1; i<=nx; i++) {
		ib=x_ijk2ib(i, j, k+1, nx, ny, nz+1, 0);

                farr[ic++] = zcornsv[4 * ib + 1 * 3 - 1];
                farr[ic++] = zcornsv[4 * ib + 1 * 4 - 1];
	    }
	}
    }

    grd3d_write_eclrecord(fc, "ZCORN", 2, &idum, farr, &ddum, nzcorn,
                          XTGDEBUG);
    free(farr);

    /*
     *-------------------------------------------------------------------------
     * ACTNUM
     *-------------------------------------------------------------------------
     */
    nact = nx * ny * nz;

    grd3d_write_eclrecord(fc, "ACTNUM", 1, p_actnum_v, &fdum,
                          &ddum, nact, XTGDEBUG);

    /*
     *-------------------------------------------------------------------------
     * ENDGRID
     *-------------------------------------------------------------------------
     */

    itmp[0] = 0;

    grd3d_write_eclrecord(fc, "ENDGRID", 1, itmp, &fdum,
                          &ddum, 1, XTGDEBUG);

    /*
     *-------------------------------------------------------------------------
     * Close file
     *-------------------------------------------------------------------------
     */

    fclose(fc);

    logger_info(LI, FI, FU, "Export to EGRID format, done!");

}
