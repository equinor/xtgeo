/*
 ******************************************************************************
 *
 * Read an Eclipse binary record based on earlier scan
 *
 ******************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"

/*
 ******************************************************************************
 *
 * NAME:
 *    grd3d_read_eclrecord.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Based on a scan, the record potition in file and length is
 *    known in advance. This reads the actual data array and returns.
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
 *    recstart         i     Start of record in file, in bytes
 *    rectype          i     Type of record to read (1=INT, 2=FLT, 3=DBL)
 *    intv             o     Preallocated Int array (if rectype is 1)
 *    nint             i     The allocated record total length
 *    floatv           o     Preallocated Float array (if rectype is 2)
 *    nflt             i     The allocated record total length
 *    doublev          o     Preallocated Double array (if rectype is 3)
 *    ndbl             i     The allocated record total length if double
 *    debug            i     Debug level
 *
 * RETURNS:
 *    Function: nelements upon success, negative if problems
 *    Resulting vectors
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ******************************************************************************
 */



int grd3d_read_eclrecord (FILE *fc, long recstart,
                          int rectype, int *intv, long nint, float *floatv,
                          long nflt, double *doublev, long ndbl,
                          int debug)
{

    char s[24] = "grd3d_read_eclrecord";

    const int FAIL = -99;
    int ic, ier, myint, ftn1, ftn2, ftn_reclen = 0, nr, swap = 0;
    float myfloat;
    double mydouble;
    long reclength = 0, nrecs, icc;

    xtgverbose(debug);

    if (x_swap_check() == 1) swap = 1;

    if (fc == NULL) xtg_error(s, "Cannot use file");

    if (rectype == 1) reclength = nint;
    if (rectype == 2) reclength = nflt;
    if (rectype == 3) reclength = ndbl;

    /* go to file position */
    if (debug > 2) xtg_speak(s, 3, "RECSTART is %ld", recstart);
    ier = fseek(fc, recstart + 24, SEEK_SET);  /* record header is 24 bytes */

    if (ier != 0) {
        xtg_error(s, "Could not set FSEEK position");
    }

    nrecs = reclength;

    icc = 0;
    if (debug > 2) xtg_speak(s, 3, "NRECS is %d", nrecs);
    while (nrecs > 0) {
        if (fread(&ftn1, 4, 1, fc) != 1) return FAIL;
        if (debug > 2) xtg_speak(s, 3, "FTN1 (pre) is %d", ftn1);
        if (swap) SWAP_INT(ftn1);

        if (debug > 2) xtg_speak(s, 3, "FTN1 (swp) is %d", ftn1);

        if (rectype == 1) ftn_reclen = (ftn1 / sizeof(int));
        if (rectype == 2) ftn_reclen = (ftn1 / sizeof(float));
        if (rectype == 3) ftn_reclen = (ftn1 / sizeof(double));

        if (ftn_reclen < nrecs) {
            nr = ftn_reclen;
        }
        else{
            nr = nrecs;
        }

        if (debug > 2) xtg_speak(s, 3, "FTN_RECLEN is %d", ftn_reclen);
        if (debug > 2) xtg_speak(s, 3, "NR is %d", nr);

        /* read actual values with in the fortran block */
        for (ic = 0; ic < nr; ic++) {
            if (debug > 2) xtg_speak(s, 3, "IC = %d of %d", ic, nr);

            if (rectype == 1) {
                ier = fread(&myint, 4, 1, fc);
                if (ier != 1) return FAIL;
                if (swap) SWAP_INT(myint);
                intv[icc++] = myint;
            }
            else if (rectype == 2) {
                ier = fread(&myfloat, 4, 1, fc);
                if (ier != 1) return FAIL;
                if (swap) SWAP_FLOAT(myfloat);
                floatv[icc++] = myfloat;
            }
            else if (rectype == 3) {
                ier = fread(&mydouble, 8, 1, fc);
                if (ier != 1) return FAIL;
                if (swap) SWAP_DOUBLE(mydouble);
                doublev[icc++] = mydouble;
            }
        }

        /* end of record integer: */
        if (fread(&ftn2, 4, 1, fc) != 1) return FAIL;
        if (swap) SWAP_INT(ftn2);
        if (ftn1 != ftn2) return FAIL;

        /* remaining record length: */
        nrecs = nrecs - ftn_reclen;
    }

    if (icc != reclength) {
        xtg_error(s, "Something is wrong with record lengths... "
                  "icc=%d, reclength=%d", icc, reclength);
        return FAIL;
    }

    return(icc);

}
