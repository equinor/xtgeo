/*
 ******************************************************************************
 *
 * Write an Eclipse input ASCII record to file
 *
 ******************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"

/*
 ******************************************************************************
 *
 * NAME:
 *    grd3d_write_eclinput.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Feed data from XTGeo to the Eclipse input deck. The XTGeo DATA
 *    stream must be in FORTRAN order for array grid data.
 *
 *    This is the format for DATA files. The input fmt is not checked
 *    for inconsistencies.
 *
 * ARGUMENTS:
 *    fc               i     Filehandle (file must be open)
 *    recname          i     Name of record to write
 *    rectype          i     Type of record to write (1=INT, 2=FLT, 3=DBL)
 *    intv             i     Input int array (if rectype is 1)
 *    floatv           i     Input Float array (if rectype is 2)
 *    doublev          i     Input double array (if rectype is 3)
 *    nrecs            i     The record total length
 *    fmt              i     Format specifier e.g. "  %8.2f", NB with col space
 *    ncolumns         i     Number of columns
 *    debug            i     Debug level
 *
 * RETURNS:
 *    Function: EXIT_SUCCESS upon success
 *
 * TODO/ISSUES/BUGS:
 *    - Some clever type cast may simplify?
 *    - Compressed formats, such as 4*0 5*1 instead of 0 0 0 0 1 1 1 1 1
 *
 * LICENCE:
 *    CF XTGeo
 ******************************************************************************
 */

int grd3d_write_eclinput (FILE *fc,
                          char *recname,
                          int rectype, int *intv, float *floatv,
                          double *doublev, long nrecs,
                          char *fmt, int ncolumns, int debug)
{

    char sbn[24] = "grd3d_write_eclinput";
    int icwrap = 0;
    long icc = 0;

    if (debug > 2) xtg_speak(sbn, 3, "Enter %s", sbn);

    fprintf(fc,"%-8s\n", recname);

    if (rectype == 1) {
        icwrap = 0;
        for (icc = 0; icc < nrecs; icc++) {
            fprintf(fc, fmt, intv[icc]);
            icwrap++;
            if (icwrap >= ncolumns) {
                fprintf(fc, "\n");
                icwrap = 0;
            }
        }
    }

    if (rectype == 2) {
        icwrap = 0;
        for (icc = 0; icc < nrecs; icc++) {
            fprintf(fc, fmt, floatv[icc]);
            icwrap++;
            if (icwrap >= ncolumns) {
                fprintf(fc, "\n");
                icwrap = 0;
            }
        }
    }

    if (rectype == 3) {
        icwrap = 0;
        for (icc = 0; icc < nrecs; icc++) {
            fprintf(fc, fmt, doublev[icc]);
            icwrap++;
            if (icwrap >= ncolumns) {
                fprintf(fc, "\n");
                icwrap = 0;
            }
        }
    }

    if (icwrap == 0) fprintf(fc, "/\n\n");
    if (icwrap > 0) fprintf(fc, "\n/\n\n");

    return EXIT_SUCCESS;
}
