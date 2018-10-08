/*
 ******************************************************************************
 *
 * Write an Eclipse binary record to file
 *
 ******************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"

/*
 ******************************************************************************
 *
 * NAME:
 *    grd3d_write_eclrecord.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Feed data from XTGeo to the Eclipse binary Fortran format (big endian).
 *    The XTGeo DATA stream must be in FORTRAN order.
 *
 *    This is the format for GRID, EGRID, INIT and restart files.
 *
 *    'INTEHEAD'         200 'INTE'
 *    -1617152669        9701           2       -2345       -2345       -2345
 *          -2345       -2345          20          15           8        1639
 *            246       -2345           7       -2345           0           8
 *              0          15           4           0           6          21
 *    ETC!....
 *
 * ARGUMENTS:
 *    fc               i     Filehandle (file must be open)
 *    recname          i     Name of record to write
 *    rectype          i     Type of record to write (1=INT, 2=FLT, 3=DBL)
 *    intv             i     Input int array (if rectype is 1)
 *    nint             i     The int record total length
 *    floatv           i     Input Float array (if rectype is 2)
 *    nflt             i     The float record total length
 *    doublev          i     Input double array (if rectype is 3)
 *    ndbl             i     The record total length if double
 *    debug            i     Debug level
 *
 * RETURNS:
 *    Function: EXIT_SUCCESS upon success
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    Statoil property
 ******************************************************************************
 */


int grd3d_write_eclrecord (FILE *fc,
                           char *recname,
                           int rectype, int *intv, long nint, float *floatv,
                           long nflt, double *doublev, long ndbl,
                           int debug)
{

    char s[24] = "grd3d_write_eclrecord";
    int ib, swap = 0;
    int myint, mylen;
    float myfloat;
    char mychar[9]="", mytype[5]="";
    long reclength = 0, nrecs;

    xtgverbose(debug);

    strncpy(mychar, recname, 8);

    if (x_swap_check() == 1) swap = 1;

    if (fc == NULL) xtg_error(s, "Cannot use file");

    if (rectype == 1) {
        reclength = nint;
        strncpy(mytype, "INTE", 4);
    }
    if (rectype == 2) {
        reclength = nflt;
        strncpy(mytype, "REAL", 4);
    }
    if (rectype == 3) {
        reclength = ndbl;
        strncpy(mytype, "DBLE", 4);
    }

    nrecs = reclength;

 /* header: */
    myint = 16;
    if (swap) SWAP_INT(myint);
    fwrite(&myint, 4, 1, fc);
    fwrite(mychar, 8, 1, fc);
    mylen = reclength; if (swap) SWAP_INT(mylen);
    fwrite(&mylen, 1, 4, fc);
    fwrite(mytype, 1, 4, fc);
    fwrite(&myint, 1, 4, fc);

    myint = nrecs;
    if (swap) SWAP_INT(myint);

    fwrite(&myint, 1, 4, fc);
    for (ib = 0; ib < nrecs; ib++) {
        myfloat = floatv[ib];
        if (swap) SWAP_INT(myfloat);
        fwrite(&myfloat, 1, 4, fc);
    }
    fwrite(&myint, 1, 4, fc);

    return EXIT_SUCCESS;
}
