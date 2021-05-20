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
 *
 * RETURNS:
 *    Void function
 *
 * LICENCE:
 *    CF. XTGeo license
 ******************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"
#include "logger.h"

void
grd3d_export_grdeclprop2(int nx,
                         int ny,
                         int nz,
                         int ptype,
                         int *p_iprop_v,
                         float *p_fprop_v,
                         double *p_dprop_v,
                         char *pname,
                         char *fmt,
                         char *filename,
                         int mode,
                         int flag)

{
    long nlen;
    FILE *fc = NULL;

    /*
     *-------------------------------------------------------------------------
     * Open file
     *-------------------------------------------------------------------------
     */

    if (mode == 0)
        logger_info(LI, FI, FU, "Opening binary GRDECL file...");
    if (mode == 1)
        logger_info(LI, FI, FU, "Opening text GRDECL file...");

    if (flag == 0)
        fc = fopen(filename, "wb");
    if (flag == 1)
        fc = fopen(filename, "ab");

    if (fc == NULL) {
        throw_exception("Could not open file: grd3d_export_grdeclprop2");
        return;
    }
    nlen = nx * ny * nz;

    if (mode == 0) {
        grd3d_write_eclrecord(fc, pname, ptype, p_iprop_v, p_fprop_v, p_dprop_v, nlen);
    } else {
        /* todo: smart analysis of values to decide formatting */
        int ncol = 6;

        grd3d_write_eclinput(fc, pname, ptype, p_iprop_v, p_fprop_v, p_dprop_v, nlen,
                             fmt, ncol);
    }

    logger_info(LI, FI, FU, "Writing prop to (B)GRDECL file... done");
    fclose(fc);
}
