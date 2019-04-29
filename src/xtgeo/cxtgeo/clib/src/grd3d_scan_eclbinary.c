/*
 ******************************************************************************
 *
 * Scan Eclipse binary files for data
 *
 ******************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"

/*
 ******************************************************************************
 *
 * NAME:
 *    grd3d_scan_eclbinary.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
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
 *    rectypes        o     An array with record types: 1 = INT, 2 = FLOAT,
 *                          3 = DOUBLE, 4 = CHAR (8), ...
 *    reclengths      o     An array with record lengths (no of elements)
 *    recstarts       o     An array with record starts (in bytes)
 *    maxkw           i     Max number of kwords (allocated length of arrays)
 *    debug           i     Debug level
 *
 * RETURNS:
 *    Function: Number of keywords read. If problems, a negative value
 *    Resulting vectors
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ******************************************************************************
 */

/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * local function(s)
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 */

int _scan_ecl_bin_record(FILE *fc, char *cname, int *cntype, long *rnlen,
                         long npos1, long *npos2, int debug)
{
    char s[24] = "_scan_ecl_bin_record";
    int swap = 0;
    int ftn, ier, doread, rlen, nbyte, nval;
    const int FAIL = -88;
    char ctype[5] = "NNNN";
    long ncum = 0;

    xtgverbose(debug);

    if (x_swap_check() == 1) swap = 1;

    /* read the description line, as e.g.:
       [<16>'CORNERS '          24 'REAL'<16>] where <16> are 4 byte int
       determining record length, here (8+4+4 = 16 bytes)
    */

    ier = fread(&ftn, 4, 1, fc);

    if (ier != 1) {
        xtg_speak(s, 2, "IER != 1, unswapped FTN and IER is %d %d EOF=%d",
                  ftn, ier, EOF);
        if (ier == 0) return EOF;
        if (ier == EOF) return EOF;
        if (ier != EOF) return FAIL;
    }

    if (swap) SWAP_INT(ftn);
    xtg_speak(s, 2, "Read FTN <%d>", ftn);


    /* read keyword, arraylength and type */
    if ((ier = fread(cname, 8, 1, fc)) != 1) return FAIL;
    cname[8]='\0';
    xtg_speak(s, 2, "Read <%s>", cname);

    if ((ier = fread(&rlen, 4, 1, fc)) != 1) return FAIL;
    if (swap) SWAP_INT(rlen);
    xtg_speak(s, 2, "Read RLEN <%d>", rlen);

    if ((ier = fread(ctype, 4, 1, fc)) != 1) return FAIL;
    ctype[4]='\0';
    xtg_speak(s, 2, "Read CTYPE <%s>", ctype);

    *cntype = -1;
    if (strcmp(ctype, "INTE") == 0) *cntype = 1;
    if (strcmp(ctype, "REAL") == 0) *cntype = 2;
    if (strcmp(ctype, "DOUB") == 0) *cntype = 3;
    if (strcmp(ctype, "CHAR") == 0) *cntype = 4;
    if (strcmp(ctype, "LOGI") == 0) *cntype = 5;
    if (strcmp(ctype, "MESS") == 0) *cntype = 6;

    if (*cntype == -1) return FAIL;

    ier = fread(&ftn, 4, 1, fc);
    if (swap) SWAP_INT(ftn);
    xtg_speak(s, 2, "<%s>: Last FTN and IER is %d %d", cname, ftn, ier);

    if (ier != 1) return FAIL;

    /*
     * Report the end byte position of this record. The challenge is that
     * there is a unknown number of Fortran records to loop,
     * as the need to be counted. So far:
     */
    ncum = npos1 + 4 + 8 + 4 + 4 + 4;  /* [ftn KEYWORD nlen TYPE ftn] */
    xtg_speak(s, 2, "NCUM is %d", ncum);
    xtg_speak(s, 2, "RLEN is %d", rlen);

    doread = rlen;
    nval = 0;
    while(doread) {
        if (fread(&ftn, 4, 1, fc) != 1) return FAIL;
        if (swap) SWAP_INT(ftn);

        if (debug>2) xtg_speak(s, 3, "Data block ftn is %d", ftn);
        nbyte = 4;
        if (*cntype > 2) nbyte = 8;
        if (*cntype == 5) nbyte = 1;
        if (*cntype == 6) nbyte = 4;  /* MESS, correct?? */

        ncum = ncum + ftn + 4 + 4;

        if (debug>2) xtg_speak(s, 3, "ncum is %d", ncum);

        if (fseek(fc, ncum, SEEK_SET) != 0) return FAIL;

        /* count used amount of the array length */
        nval += ftn/nbyte ;

        if (debug>2) xtg_speak(s, 3, "Data block nval is %d", nval);

        if (nval >= rlen) doread = 0;
    }

    *npos2 = ncum;
    *rnlen = rlen;

    xtg_speak(s, 2, "NCUM and NPOS2 is %d, %d", ncum, *npos2);

    return EXIT_SUCCESS;
}


long grd3d_scan_eclbinary (FILE *fc, char *keywords, int *rectypes,
                           long *reclengths, long *recstarts, long maxkw,
                           int debug)
{

    char s[24] = "grd3d_scan_eclbinary";
    char cname[9] = "unset";
    int ios, cntype;
    long i = 0, npos1, npos2, rnlen;
    const int FAIL = -99;
    const int _FAIL = -88;

    xtgverbose(debug);

    npos1 = 0;

    ios = 0;

    keywords[0] = '\0';

    rewind(fc);

    while (ios == 0) {
        ios = _scan_ecl_bin_record(fc, cname, &cntype, &rnlen, npos1,
                                   &npos2, debug);

        if (ios != 0) break;

        xtg_speak(s, 2, "Keyword is <%s>, type is <%d>, "
                  "RECLEN is <%ld>, npos1 and npos2: <%ld> <%ld>",
                  cname, cntype, rnlen, npos1, npos2);

        strcat(keywords, cname);
        strcat(keywords, "|");

        reclengths[i] = rnlen;
        rectypes[i] = cntype;
        recstarts[i] = npos1;

        if (i >= maxkw) {
            xtg_error(s, "Number of max keywords reached: %d", maxkw);
            return -2;
        }

        i++;
        npos1 = npos2;
    }

    if (ios == FAIL || ios == _FAIL) {
        xtg_error(s, "Unsuccessful read of file (ios = %d)", ios);
        return -1;
    }

    /* remove last | */
    keywords[strlen(keywords)-1] = 0;

    return (i);  /* return number of actual keywords */
}
