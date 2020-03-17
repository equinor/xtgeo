
/*
 ******************************************************************************
 *
 * NAME:
 *    grd3d_import_grdecl_prop.c
 *
 * DESCRIPTION:
 *    Import a grid property on Eclipse style ASCII GRDECL format. Note that
 *    keywords longer than 8 is accepted (up to 32 letters). The property will
 *    be returned in C order (while the property in the file is stored in
 *    F order).
 *
 *    Also INT numbers are read by this routine; do foat to int conversion in
 *    Python.
 *
 * ARGUMENTS:
 *    filename       i     File name to import from
 *    nx, ny, nz     i     Grid dimensions
 *    pname          i     Property name
 *    p_prop_v      i/o    Pointer array to update
 *    nlen           i     Length of property (nx ny nz), SWIG usage
 *    option         i     Options flag for later usage
 *    debug          i     Debug level
 *
 * RETURNS:
 *    0 if success, -1 if keyword not found. Pointer is updated.
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ******************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"
#include "logger.h"

int
grd3d_import_grdecl_prop(char *filename,
                         int ncol,
                         int nrow,
                         int nlay,
                         char *pname,
                         double *p_prop_v,
                         long nlen,
                         int option)

{
    char cname[33];
    long ic, line;
    int nchar, found = 0;
    int icol, jrow, klay;
    double fvalue;
    FILE *fc;

    logger_info(LI, FI, FU, "Import Property on Eclipse GRDECL format ...");

    fc = fopen(filename, "rb");

    nchar = strlen(pname);

    for (line = 1; line < 99999999; line++) {

        /* Get word */
        if (fgets(cname, 33, fc) == NULL) {
            fclose(fc);
            return -1;
        }

        if (strncmp(cname, pname, nchar) == 0) {
            found = 1;
            for (klay = 1; klay <= nlay; klay++) {
                for (jrow = 1; jrow <= nrow; jrow++) {
                    for (icol = 1; icol <= ncol; icol++) {
                        if (fscanf(fc, "%lf", &fvalue) != 1)
                            logger_error(LI, FI, FU, "Error in reading %s", pname);

                        /* map directly to C order */
                        ic = x_ijk2ic(icol, jrow, klay, ncol, nrow, nlay, 0);
                        p_prop_v[ic] = fvalue;
                    }
                }
            }
            break;
        }
    }

    fclose(fc);

    if (found == 0)
        return -1;

    return EXIT_SUCCESS;
}
