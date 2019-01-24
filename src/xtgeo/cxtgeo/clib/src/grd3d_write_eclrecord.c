/*
 ******************************************************************************
 *
 * Write an Eclipse binary record to file
 *
 ******************************************************************************
 */

#include <limits.h>
#include "libxtg.h"
#include "libxtg_.h"

/*
 ******************************************************************************
 *
 * NAME:
 *    grd3d_write_eclrecord.c
 *
 * DESCRIPTION:
 *    Feed data from XTGeo to the Eclipse binary Fortran format (big endian).
 *    The XTGeo data arrays must be in FORTRAN order.
 *
 *    This is the format for binary GRDECL, GRID, EGRID, INIT and restart
 *    files (and also summary files). The layout is:
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
 *    Cf. XTGeo license
 ******************************************************************************
 */


int grd3d_write_eclrecord (FILE *fc, char *recname,
                           int rectype, int *intv, long nint, float *floatv,
                           long nflt, double *doublev, long ndbl,
                           int debug)
{

    int ib, swap = 0;
    int myint, myint2, mylen, nbyte;
    float myfloat;
    double mydouble;
    char mychar[8]="", mytype[4]="";
    long nrecs = 0;

    char sbn[24] = "grd3d_write_eclrecord";
    xtgverbose(debug);

    sprintf(mychar, "%-8s", recname);
    // u_eightletter(mychar);

    if (x_swap_check() == 1) swap = 1;

    if (fc == NULL) xtg_error(sbn, "Cannot use file, file descriptor is NULL");

    if (rectype == 1) {
        nrecs = nint;
        nbyte = 4;
        strncpy(mytype, "INTE", 4);
    }
    if (rectype == 2) {
        nrecs = nflt;
        nbyte = 4;
        strncpy(mytype, "REAL", 4);
    }
    if (rectype == 3) {
        nrecs = ndbl;
        nbyte = 8;
        strncpy(mytype, "DBLE", 4);
    }

 /* header: */
    myint = 16;
    if (swap) SWAP_INT(myint);
    fwrite(&myint, 4, 1, fc);
    fwrite(mychar, 1, 8, fc);
    mylen = nrecs;
    if (swap) SWAP_INT(mylen);
    fwrite(&mylen, 4, 1, fc);
    fwrite(mytype, 1, 4, fc);
    fwrite(&myint, 4, 1, fc);

    if (nrecs * nbyte > INT_MAX) {
        xtg_error(sbn, "Record size > %d; not supported (yet)", INT_MAX);
    }

    myint2 = nrecs * nbyte;
    xtg_speak(sbn, 2, "NRECS %d (total bytes %d) for <%s> of type <%s>",
              nrecs, myint2, mychar, mytype);
    if (swap) SWAP_INT(myint2);

    fwrite(&myint2, 4, 1, fc);
    for (ib = 0; ib < nrecs; ib++) {
        if (rectype == 1) {
            myint = intv[ib];
            if (swap) SWAP_INT(myint);
            fwrite(&myint, 4, 1, fc);
        }
        else if (rectype == 2) {
            myfloat = floatv[ib];
            if (swap) SWAP_FLOAT(myfloat);
            fwrite(&myfloat, 4, 1, fc);
        }
        else if (rectype == 3) {
            mydouble = doublev[ib];
            if (swap) SWAP_DOUBLE(mydouble);
            fwrite(&mydouble, 8, 1, fc);
        }
    }
    fwrite(&myint2, 4, 1, fc);

    return EXIT_SUCCESS;
}
