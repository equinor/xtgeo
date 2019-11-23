/*
 ******************************************************************************
 *
 * Reports the Eclipse restart TSTEPS in UNRST like files
 *
 ******************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"

/*
 ******************************************************************************
 *
 * NAME:
 *    grd3d_ecl_tsteps.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *
 * ARGUMENTS:
 *    efile            i     Filename to read from
 *    seqnum, day ...  o     vectors
 *    nmax             i     Max number of dates
 *    debug            i     debug level
 * RETURNS:
 *    Function: nelements upon success, 0 or negative if problems
 *    Resulting vectors
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ******************************************************************************
 */

/* since windows is missing strsep() */
char* _mystrsep(char **stringp, const char *delim)
{

    char *start = *stringp, *p = start ? strpbrk(start, delim) : NULL;

    if (!p) {
        *stringp = NULL;
    } else {
        *p = 0;
        *stringp = p + 1;
    }

    return start;
}

int grd3d_ecl_tsteps (FILE *fc, int *seqnums, int *day, int *mon, int *year,
                      int nmax, int debug)
{

    char s[24] = "grd3d_ecl_tsteps";

    char *keywords;
    int *rectypes;
    long *reclengths, *recstarts;

    char *token, *tofree;
    int ic, nc, nkeys, keytype;
    long rlen, rstart, nvals;
    int *intrecord;
    float *xfloat = NULL;
    double *xdouble = NULL;

    int maxkw = 1000000;

    keywords = (char *) calloc(maxkw*10, sizeof(char));
    rectypes = (int *) calloc(maxkw, sizeof(int));
    reclengths = (long *) calloc(maxkw, sizeof(long));
    recstarts = (long *) calloc(maxkw, sizeof(long));

    xtgverbose(debug);

    rewind(fc);

    /* do scan */
    nkeys = grd3d_scan_eclbinary(fc, keywords, rectypes, reclengths,
                                 recstarts, maxkw, debug);

    /* now look for SEQNUM and INTEHEADs */
    tofree = keywords;
    ic = 0;
    nc = 0;
    //    while ((token = strsep(&keywords, "|")) != NULL) {
    while ((token = _mystrsep(&keywords, "|")) != NULL) {

        if (strcmp(token, "SEQNUM  ") == 0) {
            keytype = 1;
            rlen = reclengths[ic];
            rstart = recstarts[ic];
            intrecord = calloc(rlen, sizeof(int));  /* knows this is int */
            nvals = grd3d_read_eclrecord(fc, rstart, keytype,
                                         intrecord, rlen, xfloat, 0,
                                         xdouble, 0);

            seqnums[nc] = intrecord[0];

            free(intrecord);

        }

        if (strcmp(token, "INTEHEAD") == 0) {
            keytype = 1;
            rlen = reclengths[ic];
            rstart = recstarts[ic];
            intrecord = calloc(rlen, sizeof(int));  /* knows this is int */
            nvals = grd3d_read_eclrecord(fc, rstart, keytype,
                                         intrecord, rlen, xfloat, 0,
                                         xdouble, 0);

            day[nc] = intrecord[64];
            mon[nc] = intrecord[65];
            year[nc] = intrecord[66];

            free(intrecord);

            nc++;

            if (nc >= nmax) {
                xtg_error(s, "Fail in dimensions in %s", s);
            }
        }

        ic++;
    }

    free(tofree);

    free(keywords);
    free(rectypes);
    free(reclengths);
    free(recstarts);

    return(nc);
}
