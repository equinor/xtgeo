/*
 ***************************************************************************************
 *
 * NAME:
 *    grd3d_scan_eclbinary.c
 *
 *
 * DESCRIPTION:
 *    Quick scan Eclipse output (GRID/EGRID/INIT/UNRST...) and return:
 *    * A list of keywords
 *    * A list of keywords types
 *    * A list of start positions (in bytes) for data *
 *    * A list of array lengths
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
 *    For UNIFIED restart files, the timestep is indicated by a SEQNUM keyword:
 *
 *    'SEQNUM  '           1 'INTE'
 *              0
 *    The actual DATE is items no  65, 66, 67 within INTEHEAD (counted from 1)
 *
 *    For the binary form, the record starts and ends with a 4 byte integer,
 *    that says how long the current record is, in bytes (Fortran style).
 *
 * ARGUMENTS:
 *    fc              i     Filehandle (stream) to read from
 *    keywords        o     A *char where the keywords are separated by a |
 *    nkey.._x_let..  i     No. of kwords * letters (10 letters per keyword) > SWIG
 *    rectypes        o     An array with record types: 1 = INT, 2 = FLOAT,
 *                             3 = DOUBLE, 4 = CHAR (8), ...
 *    nrectypes       i     For SWIG interface
 *    reclengths      o     An array with record lengths (no of elements)
 *    nreclengths     i     For SWIG interface
 *    recstarts       o     An array with record starts (in bytes)
 *    nrecstarts      i     For SWIG allocation
 *
 * RETURNS:
 *    Function: Number of keywords read. If problems, a negative value
 *    Resulting vectors
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ***************************************************************************************
 */
#include "libxtg.h"
#include "libxtg_.h"
#include "logger.h"
#include <limits.h>
/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * local function(s)
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */

int
_scan_ecl_bin_record(FILE *fc,
                     char *cname,
                     int *cntype,
                     long *rnlen,
                     long npos1,
                     long *npos2)
{
    int swap = 0;
    int ftn, ier, doread, rlen, nbyte, nval;
    const int FAIL = -88;
    char ctype[5] = "NNNN";
    long ncum = 0;

    if (x_swap_check() == 1)
        swap = 1;

    /* read the description line, as e.g.:
       [<16>'CORNERS '          24 'REAL'<16>] where <16> are 4 byte int
       determining record length, here (8+4+4 = 16 bytes)
    */

    ier = fread(&ftn, 4, 1, fc);

    if (ier != 1) {
        if (ier == 0)
            return EOF;
        if (ier == EOF)
            return EOF;
        if (ier != EOF)
            return FAIL;
    }

    if (swap)
        SWAP_INT(ftn);

    /* read keyword, arraylength and type */
    if ((ier = fread(cname, 8, 1, fc)) != 1)
        return FAIL;
    cname[8] = '\0';

    if ((ier = fread(&rlen, 4, 1, fc)) != 1)
        return FAIL;
    if (swap)
        SWAP_INT(rlen);

    if ((ier = fread(ctype, 4, 1, fc)) != 1)
        return FAIL;
    ctype[4] = '\0';

    *cntype = -1;
    if (strcmp(ctype, "INTE") == 0)
        *cntype = 1;
    if (strcmp(ctype, "REAL") == 0)
        *cntype = 2;
    if (strcmp(ctype, "DOUB") == 0)
        *cntype = 3;
    if (strcmp(ctype, "CHAR") == 0)
        *cntype = 4;
    if (strcmp(ctype, "LOGI") == 0)
        *cntype = 5;
    if (strcmp(ctype, "MESS") == 0)
        *cntype = 6;

    if (*cntype == -1)
        return FAIL;

    ier = fread(&ftn, 4, 1, fc);
    if (swap)
        SWAP_INT(ftn);

    if (ier != 1)
        return FAIL;

    /*
     * Report the end byte position of this record. The challenge is that
     * there is a unknown number of Fortran records to loop,
     * as the need to be counted. So far:
     */
    ncum = npos1 + 4 + 8 + 4 + 4 + 4; /* [ftn KEYWORD nlen TYPE ftn] */

    doread = rlen;
    nval = 0;
    while (doread) {
        if (fread(&ftn, 4, 1, fc) != 1)
            return FAIL;
        if (swap)
            SWAP_INT(ftn);

        nbyte = 4;
        if (*cntype > 2)
            nbyte = 8;
        if (*cntype == 5)
            nbyte = 1;
        if (*cntype == 6)
            nbyte = 4; /* MESS, correct?? */

        ncum = ncum + ftn + 4 + 4;

        if (fseek(fc, ncum, SEEK_SET) != 0)
            return FAIL;

        /* count used amount of the array length */
        nval += ftn / nbyte;

        if (nval >= rlen)
            doread = 0;
    }

    *npos2 = ncum;
    *rnlen = rlen;

    return EXIT_SUCCESS;
}

long
grd3d_scan_eclbinary(FILE *fc,
                     char *keywords,
                     int nkeywords_x_letters,
                     int *rectypes,
                     long nrectypes,
                     long *reclengths,
                     long nreclengths,
                     long *recstarts,
                     long nrecstarts)
{

    char cname[9] = "unset";
    int ios, cntype;
    long i = 0, npos1, npos2, rnlen;
    const int FAIL = -99;
    const int _FAIL = -88;

    if (fc == NULL) {
        throw_exception("Unrecoverable error, NULL file pointer received "
                        "(grd3d_scan_eclbinary)");
        return EXIT_FAILURE;
    }

    if (nkeywords_x_letters > INT_MAX) {
        throw_exception("Unrecoverable error, number of requested keyword letters "
                        "exceeds system limit (grd3d_scan_eclbinary)");
        return -3;
    }

    npos1 = 0;
    ios = 0;
    keywords[0] = '\0';
    rewind(fc);

    if (nkeywords_x_letters > INT_MAX) {
        throw_exception("Unreverable error, number of requested keyword letters "
                        "exceeds system limit (grd3d_scan_eclbinary)");
        return -3;
    }

    while (ios == 0) {
        ios = _scan_ecl_bin_record(fc, cname, &cntype, &rnlen, npos1, &npos2);

        if (ios != 0)
            break;
        strcat(keywords, cname);
        strcat(keywords, "|");

        reclengths[i] = rnlen;
        rectypes[i] = cntype;
        recstarts[i] = npos1;

        if (i >= nreclengths) {
            /* treat this return code in the Python client */
            return -2;
        }

        i++;
        npos1 = npos2;
    }

    if (ios == FAIL || ios == _FAIL) {
        throw_exception("Error in reading Eclipse file (grd_scan_eclbinary)");
        return -1;
    }

    /* remove last | */
    keywords[strlen(keywords) - 1] = 0;

    return (i); /* return number of actual keywords */
}
