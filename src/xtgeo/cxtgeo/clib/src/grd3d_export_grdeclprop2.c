/*
 ******************************************************************************
 *
 * Export to GRDECL format for grid properties
 *
 ******************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"

/*
 ******************************************************************************
 *
 * NAME:
 *    grd3d_export_grdeclprops2.c
 *
 * DESCRIPTION:
 *    Export to Eclipse GRDECL format, either ASCII text or Ecl binary style
 *
 * ARGUMENTS:
 *    nx, ny, nz     i     NCOL, NROW, NLAY
 *    p_prop_v       i     PROP array
 *    ntot           i     Array length total (for SWIG)
 *    ptype          i     Property type for storage, 1=INT, 2=FLOAT, 3=DOUBLE
 *    pname          i     Name of property
 *    pfile          i     File name
 *    mode           i     File mode, 1 ascii, 0  is binary
 *    flag           i     0: new file, 1 append
 *    debug          i     Debug level
 *
 * RETURNS:
 *    Void function
 *
 * LICENCE:
 *    CF. XTGeo license
 ******************************************************************************
 */


void grd3d_export_grdeclprop2 (
                               int nx,
                               int ny,
                               int nz,
                               int ptype,
                               int *p_iprop_v,
                               float *p_fprop_v,
                               double *p_dprop_v,
                               char *pname,
                               char *filename,
                               int mode,
                               int flag,
                               int debug
                               )

{
    long nlen;
    FILE *fc = NULL;

    char sbn[24] = "grd3d_exp..grdeclprop2";
    xtgverbose(debug);

    xtg_speak(sbn, 2,"Enter %s", sbn);

    /*
     *-------------------------------------------------------------------------
     * Open file
     *-------------------------------------------------------------------------
     */

    if (mode == 0) xtg_speak(sbn, 2,"Opening binary GRDECL file...");
    if (mode == 1) xtg_speak(sbn, 2,"Opening text GRDECL file...");

    if (flag == 0) fc = fopen(filename, "wb");
    if (flag == 1) fc = fopen(filename, "ab");

    if (fc == NULL) xtg_error(sbn, "Cannot open file!");


    xtg_speak(sbn, 2,"Exporting property %s", pname);

    nlen = nx * ny * nz;

    if (mode == 0) {
        grd3d_write_eclrecord(fc, pname, ptype, p_iprop_v, p_fprop_v,
                              p_dprop_v, nlen, debug);
    }
    else{
        /* todo: smart analysis of values to decide formatting */
        char fmt[10] = " %8d";
        int ncol = 6;
        if (ptype > 1) {
            strcpy(fmt, " %13.4f");
        }

        grd3d_write_eclinput(fc, pname, ptype, p_iprop_v, p_fprop_v,
                             p_dprop_v, nlen, fmt, ncol, debug);
    }

    fclose(fc);
}
