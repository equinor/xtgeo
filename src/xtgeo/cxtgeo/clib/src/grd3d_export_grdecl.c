/*
 ******************************************************************************
 *
 * Export to GRDECL foramt for grid geometry
 *
 ******************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"

/*
 ******************************************************************************
 *
 * NAME:
 *    grd3d_export_grdecl.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Export to Eclipse GRDECL format
 *
 * ARGUMENTS:
 *    nx, ny, nz     i     NCOL, NROW, NLAY
 *    p_coord_v      i     COORD array
 *    p_zcorn_v      i     ZCORN array
 *    p_actnum_v     i     ACTNUM array
 *    gfile          i     File name
 *    mode           i     File mode, 1 ascii, 0  is binary
 *    debug          i     Debug level
 *
 * RETURNS:
 *    Void function
 *
 * LICENCE:
 *    CF. XTGeo license
 ******************************************************************************
 */


void grd3d_export_grdecl (
			  int nx,
			  int ny,
			  int nz,
			  double *p_coord_v,
			  double *p_zcorn_v,
			  int *p_actnum_v,
			  char *filename,
                          int mode,
			  int debug
			  )

{
    int i, j, k, jj;
    long ic = 0, ib = 0;
    FILE *fc;
    char sbn[24] = "grd3d_export_grdecl";
    int idum;
    long ncc, ncoord, nzcorn, nact;
    float *farr, fdum;
    double ddum;

    xtgverbose(debug);

    xtg_speak(sbn, 2,"==== Entering grd3d_export_grdecl ====");

    /*
     *-------------------------------------------------------------------------
     * Open file
     *-------------------------------------------------------------------------
     */

    if (mode == 0) xtg_speak(sbn, 2,"Opening binary GRDECL file...");
    if (mode == 1) xtg_speak(sbn, 2,"Opening text GRDECL file...");

    fc = fopen(filename, "wb"); /* The b will ensure Unix style ASCII on win */
    if (fc == NULL) xtg_error(sbn, "Cannot open file!");

    /*
     *-------------------------------------------------------------------------
     * SPECGRID
     *-------------------------------------------------------------------------
     */

    xtg_speak(sbn, 2, "Exporting SPECGRID...");
    int itmp[4]; itmp[0] = nx; itmp[1] = ny; itmp[2] = nz; itmp[3] = 1;

    if (mode == 0) {
        grd3d_write_eclrecord(fc, "SPECGRID", 1, itmp, 4, &fdum, 0,
                              &ddum, 0, debug);
    }
    else{
        grd3d_write_eclinput(fc, "SPECGRID", 1, itmp, 4, &fdum, 0,
                             &ddum, 0, "  %5d", 10, debug);
    }

    /*
     *-------------------------------------------------------------------------
     * COORD
     *-------------------------------------------------------------------------
     */
    xtg_speak(sbn, 2, "Exporting COORD...");
    ncoord = (nx + 1) * (ny + 1) * 6;
    farr = calloc(ncoord, sizeof(float));

    ib=0;
    ncc = 0;
    for (j = 0; j <= ny; j++) {
	for (i = 0;i <= nx; i++) {

            for (jj = 0; jj < 6; jj++) farr[ncc++] = p_coord_v[ib + jj];

            ib = ib + 6;
        }
    }

    if (mode == 0) {
        grd3d_write_eclrecord(fc, "COORD", 2, &idum, 0, farr, ncoord, &ddum, 0,
                              debug);
    }
    else{
        grd3d_write_eclinput(fc, "COORD", 2, &idum, 0, farr, ncoord, &ddum, 0,
                             "  %15.3f", 6, debug);
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

    xtg_speak(sbn, 2, "Exporting ZCORN...");

    nzcorn = nx * ny * nz * 8;
    farr = calloc(nzcorn, sizeof(float));
    for (k = 1; k <= nz; k++) {
	/* top */
	for (j = 1; j <= ny; j++) {
	    for (i = 1; i <= nx; i++) {
		ib = x_ijk2ib(i, j, k, nx, ny, nz + 1, 0);

                farr[ic++] = p_zcorn_v[4 * ib + 1 * 1 - 1];
                farr[ic++] = p_zcorn_v[4 * ib + 1 * 2 - 1];
	    }

	    for (i = 1; i <= nx; i++) {
		ib = x_ijk2ib(i, j, k, nx, ny, nz + 1, 0);

                farr[ic++] = p_zcorn_v[4 * ib + 1 * 3 - 1];
                farr[ic++] = p_zcorn_v[4 * ib + 1 * 4 - 1];
	    }
	}

        /* bottom */
	for (j = 1; j <= ny; j++) {
	    for (i=1; i<=nx; i++) {
		ib=x_ijk2ib(i, j, k+1, nx, ny, nz+1, 0);

                farr[ic++] = p_zcorn_v[4 * ib + 1 * 1 - 1];
                farr[ic++] = p_zcorn_v[4 * ib + 1 * 2 - 1];
	    }
	    for (i=1; i<=nx; i++) {
		ib=x_ijk2ib(i, j, k+1, nx, ny, nz+1, 0);

                farr[ic++] = p_zcorn_v[4 * ib + 1 * 3 - 1];
                farr[ic++] = p_zcorn_v[4 * ib + 1 * 4 - 1];
	    }
	}
    }

    if (mode == 0) {
        grd3d_write_eclrecord(fc, "ZCORN", 2, &idum, 0, farr, nzcorn, &ddum, 0,
                              debug);
    }
    else {
        grd3d_write_eclinput(fc, "ZCORN", 2, &idum, 0, farr, nzcorn, &ddum, 0,
                              "  %11.3f", 6, debug);
    }
    free(farr);

    /*
     *-------------------------------------------------------------------------
     * ACTNUM
     *-------------------------------------------------------------------------
     */
    xtg_speak(sbn, 2,"Exporting ACTNUM...");

    nact = nx * ny * nz;

    if (mode == 0) {
        grd3d_write_eclrecord(fc, "ACTNUM", 1, p_actnum_v, nact, &fdum, 0,
                              &ddum, 0, debug);
    }
    else{
        grd3d_write_eclinput(fc, "ACTNUM", 1, p_actnum_v, nact, &fdum, 0,
                             &ddum, 0, "  %1d", 12, debug);
    }

    fclose(fc);
}
