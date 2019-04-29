/*
 *******************************************************************************
 *
 * Import a property on GRDECL ASCII format
 *
 *******************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"

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

int grd3d_import_grdecl_prop(
                             char *filename,
                             int ncol,
                             int nrow,
                             int nlay,
                             char *pname,
                             double *p_prop_v,
                             long nlen,
                             int option,
                             int debug
                             )

{
    char cname[33];
    long ic, line;
    int nchar, found = 0;
    int icol, jrow, klay;
    double fvalue;
    FILE *fc;

    char sbn[24] = "grd3d_imp..grdecl_prop";
    xtgverbose(debug);

    xtg_speak(sbn,2,"Import Property on Eclipse GRDECL format ...");

    xtg_speak(sbn, 2, "Opening GRDECL file...");
    fc = fopen(filename, "r");
    if (fc == NULL) xtg_error(sbn,"Cannot open file!");
    xtg_speak(sbn, 2, "Opening file...OK!");

    nchar = strlen(pname);

    for (line = 1; line < 99999999; line++) {

	/* Get word */
	if (fgets(cname, 33, fc) == NULL) return -1;

	if (strncmp(cname, pname, nchar) == 0) {
	    xtg_speak(sbn, 2, "Keyword found");
            found = 1;
	    for (klay = 1; klay <= nlay; klay++) {
                for (jrow = 1; jrow <= nrow; jrow++) {
                    for (icol = 1; icol <= ncol; icol++) {
                        if (fscanf(fc, "%lf", &fvalue) != 1)
                            xtg_error(sbn,"Error in reading %s", pname);

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

    if (found == 0) return -1;

    return EXIT_SUCCESS;
}
