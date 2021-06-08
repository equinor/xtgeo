/*
****************************************************************************************
*
* NAME:
*    grd3d_import_grdecl.c
*
* DESCRIPTION:
*    Import a grid on eclipse GRDECL format.
*
* ARGUMENTS:
*    fc             i     File handler
*    nx, ny, nz     i     Dimensions
*    coordsv       i/o    Coordinate vector (xtgeo fmt)
*    zcornsv       i/o    ZCORN vector (xtgeo fmt)
*    actnumsv      i/o    ACTNUM vector (xtgeo fmt)
*    nact           o     Number of active cells
*
* RETURNS:
*    Void, update pointer arrays and nact
*
* TODO/ISSUES/BUGS:
*    Improve this routine in general, e.g. when breaking the read loop
*
* LICENCE:
*    CF XTGeo's LICENSE
***************************************************************************************
*/
#include "libxtg.h"
#include "libxtg_.h"
#include "logger.h"

void
grd3d_import_grdecl(FILE *fc,
                    int nx,
                    int ny,
                    int nz,
                    double *coordsv,
                    long ncoord,
                    double *zcornsv,
                    long nzcorn,
                    int *actnumsv,
                    long nactnum,
                    int *nact)

{
    char cname[9];
    int i, j, k, kk, ib, line, nnx, nny, nnz, num_cornerlines, kzread;
    int ix, jy, kz, ier, nn;
    int nfact = 0, nfzcorn = 0, nfcoord = 0;
    double fvalue, fvalue1, fvalue2, xmin, xmax, ymin, ymax;
    double x1, y1, x2, y2, x3, y3, cx, cy;
    int dvalue, mamode;

    /* initial settings of data (used if MAPAXES)*/
    xmin = VERYLARGEFLOAT;
    ymin = VERYLARGEFLOAT;
    xmax = -1 * VERYLARGEFLOAT;
    ymax = -1 * VERYLARGEFLOAT;

    /* this is just a COORD sort of counter; track if X, Y, Z is read (for xmin etc)*/
    ix = 1;
    jy = 0;
    kz = 0;

    mamode = 0;

    /*
     *=================================================================================
     * Loop file... It is NOT necessary to do many tests; that should be done
     * by the calling PERL script?
     *=================================================================================
     */

    num_cornerlines = 2 * 3 * (nx + 1) * (ny + 1);

    for (line = 1; line < 9999999; line++) {

        /* Get offsets */
        if (fgets(cname, 9, fc) != NULL)
            logger_info(LI, FI, FU, "CNAME is: %s", cname);

        if (strncmp(cname, "SPECGRID", 8) == 0) {

            ier = fscanf(fc, "%d %d %d", &nnx, &nny, &nnz);
            if (ier != 3) {
                logger_error(LI, FI, FU, "Error in reading SPECGRID");
            }
        }

        if (strncmp(cname, "MAPAXES", 7) == 0) {
            ier = fscanf(fc, "%lf %lf %lf %lf %lf %lf", &x1, &y1, &x2, &y2, &x3, &y3);
            if (ier != 6) {
                logger_error(LI, FI, FU, "Error in reading MAPAXES");
            }
            mamode = 1;
        }

        if (strncmp(cname, "COORD", 5) == 0) {
            nfcoord = 1;

            for (i = 0; i < num_cornerlines; i++) {
                if (fscanf(fc, "%lf", &fvalue) != 1)
                    logger_error(LI, FI, FU, "Error in reading COORD");
                if (fvalue == 9999900.0000) {
                    fvalue = -9999.99;
                }
                coordsv[i] = fvalue;

                if (ix == 1) {
                    if (coordsv[i] < xmin)
                        xmin = coordsv[i];
                    if (coordsv[i] > xmax)
                        xmax = coordsv[i];
                    ix = 0;
                    jy = 1;
                    kz = 0;
                } else if (jy == 1) {
                    if (coordsv[i] < ymin)
                        ymin = coordsv[i];
                    if (coordsv[i] > ymax)
                        ymax = coordsv[i];
                    ix = 0;
                    jy = 0;
                    kz = 1;
                } else {
                    ix = 1;
                    jy = 0;
                    kz = 0;
                }
            }
        }

        /*
         * ZCORN: Eclipse has 8 corners pr cell, while XTGeo format
         * use 4 corners (top of cell) for NZ+1 cell. This may cause
         * problems if GAPS in GRDECL format (like BRILLD test case)
         *
         */

        if (strncmp(cname, "ZCORN", 5) == 0) {
            nfzcorn = 1;

            ib = 0;
            kzread = 0;
            kk = 0;
            for (k = 1; k <= 2 * nz; k++) {
                if (kzread == 0) {
                    kzread = 1;
                } else {
                    kzread = 0;
                }
                if (k == 2 * nz && kzread == 0)
                    kzread = 1;
                if (kzread == 1) {
                    kk += 1;
                }
                for (j = 1; j <= ny; j++) {
                    /* "left" cell margin */
                    for (i = 1; i <= nx; i++) {
                        if (fscanf(fc, "%lf", &fvalue1) != 1)
                            logger_error(LI, FI, FU, "Error in reading ZCORN");
                        if (fscanf(fc, "%lf", &fvalue2) != 1)
                            logger_error(LI, FI, FU, "Error in reading ZCORN");

                        ib = x_ijk2ib(i, j, kk, nx, ny, nz + 1, 0);
                        if (ib < 0) {
                            throw_exception("Loop resulted in index outside "
                                            "boundary in grd3d_import_grdecl");
                            return;
                        }
                        if (kzread == 1) {
                            zcornsv[4 * ib + 1 * 1 - 1] = fvalue1;
                            zcornsv[4 * ib + 1 * 2 - 1] = fvalue2;
                        }
                    }
                    /* "right" cell margin */
                    for (i = 1; i <= nx; i++) {
                        if (fscanf(fc, "%lf", &fvalue1) != 1)
                            logger_error(LI, FI, FU, "Error in reading ZCORN");
                        if (fscanf(fc, "%lf", &fvalue2) != 1)
                            logger_error(LI, FI, FU, "Error in reading ZCORN");
                        ib = x_ijk2ib(i, j, kk, nx, ny, nz + 1, 0);
                        if (ib < 0) {
                            throw_exception("Loop resulted in index outside "
                                            "boundary in grd3d_import_grd_ecl");

                            return;
                        }

                        if (kzread == 1) {
                            zcornsv[4 * ib + 1 * 3 - 1] = fvalue1;
                            zcornsv[4 * ib + 1 * 4 - 1] = fvalue2;
                        }
                    }
                }
            }
        }

        nn = 0;
        if (strncmp(cname, "ACTNUM", 6) == 0) {
            nfact = 1;
            ib = 0;
            for (k = 1; k <= nz; k++) {
                for (j = 1; j <= ny; j++) {
                    for (i = 1; i <= nx; i++) {
                        if (fscanf(fc, "%d", &dvalue) == 1) {
                            actnumsv[ib++] = dvalue;
                            if (dvalue == 1)
                                nn++;
                        } else {
                            logger_error(LI, FI, FU, "Error in reading...");
                        }
                    }
                }
            }
        }

        if (nfact == 1 && nfzcorn == 1 && nfcoord == 1) {
            break;
        }
    }

    *nact = nn;

    /* convert from MAPAXES, if present */
    if (mamode == 1) {
        for (ib = 0; ib < (nx + 1) * (ny + 1) * 6; ib = ib + 3) {
            cx = coordsv[ib];
            cy = coordsv[ib + 1];
            x_mapaxes(mamode, &cx, &cy, x1, y1, x2, y2, x3, y3, 0);
            coordsv[ib] = cx;
            coordsv[ib + 1] = cy;
        }
    }
}
