/*
 ***************************************************************************************
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
 *    floatv           i     Input Float array (if rectype is 2)
 *    doublev          i     Input double array (if rectype is 3)
 *    nrecs            i     The record total length
 *    debug            i     Debug level
 *
 * RETURNS:
 *    Function: EXIT_SUCCESS upon success
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    Cf. XTGeo license
 ***************************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"
#include <limits.h>

int
grd3d_write_eclrecord(FILE *fc,
                      char *recname,
                      int rectype,
                      int *intv,
                      float *floatv,
                      double *doublev,
                      long nrecs)
{

    int ib, swap = 0;
    int myint, mylen, nbyte;
    float myfloat;
    double mydouble;
    char mychar[9] = "", mytype[5] = "";
    int nb = 0, nmax = 0, ic = 0, im = 0, nchunk = 0, ftn = 0, nn = 0;

    sprintf(mychar, "%-8s", recname);

    if (x_swap_check() == 1)
        swap = 1;

    if (fc == NULL) {
        throw_exception("Could not open file in: grd3d_write_eclrecord");
        return EXIT_FAILURE;
    }
    if (rectype == 1) {
        nbyte = 4;
        strncpy(mytype, "INTE", 4);
    }
    if (rectype == 2) {
        nbyte = 4;
        strncpy(mytype, "REAL", 4);
    }
    if (rectype == 3) {
        nbyte = 8;
        strncpy(mytype, "DBLE", 4);
    }

    /* header: */
    myint = 16;
    if (swap)
        SWAP_INT(myint);
    fwrite(&myint, 4, 1, fc);
    fwrite(mychar, 1, 8, fc); /* 8 or 9? */
    mylen = nrecs;
    if (swap)
        SWAP_INT(mylen);
    fwrite(&mylen, 4, 1, fc);
    fwrite(mytype, 1, 4, fc);
    fwrite(&myint, 4, 1, fc);

    /* the output is written in block chunks, where each chunk <= 4000 bytes */
    /* for 4 byte entries: 1000, for 8 byte: 500 entries */

    nmax = 4000 / nbyte;
    ib = 0;
    nb = nrecs; /* remaining */

    nchunk = 1 + nrecs / nmax;
    for (im = 0; im < nchunk; im++) {
        ftn = nmax * nbyte;
        if (nb == 0)
            break;
        if (nb < nmax) {
            ftn = nb * nbyte;
            nn = nb;
        } else {
            nn = nmax;
        }

        if (swap)
            SWAP_INT(ftn);
        fwrite(&ftn, 4, 1, fc);

        for (ic = 0; ic < nn; ic++) {

            if (rectype == 1) {
                myint = intv[ib];
                if (myint > UNDEF_INT_LIMIT)
                    myint = 0;
                if (swap)
                    SWAP_INT(myint);
                fwrite(&myint, 4, 1, fc);
            } else if (rectype == 2) {
                myfloat = floatv[ib];
                if (myfloat > UNDEF_LIMIT)
                    myfloat = 0.0;
                if (swap)
                    SWAP_FLOAT(myfloat);
                fwrite(&myfloat, 4, 1, fc);
            } else if (rectype == 3) {
                mydouble = doublev[ib];
                if (mydouble > UNDEF_LIMIT)
                    mydouble = 0.0;
                if (swap)
                    SWAP_DOUBLE(mydouble);
                fwrite(&mydouble, 8, 1, fc);
            }
            ib++;
            nb--;
        }
        fwrite(&ftn, 4, 1, fc);
    }
    return EXIT_SUCCESS;
}
