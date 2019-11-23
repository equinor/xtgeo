/*
 *******************************************************************************
 *
 * Scan ROFF binary files for data
 *
 *******************************************************************************
 */

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "libxtg.h"
#include "libxtg_.h"

/*
 ******************************************************************************
 *
 * NAME:
 *    grd3d_scan_roffbinary.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    This is a new line of ROFF handling function (from 2018). Here is a
 *    quick scan ROFF binary output and return for example:
 *
 *    NameEntry              ByteposData      LenData     Datatype
 *    scale!xscale           94               1           2 (=float)
 *    zvalues!splitEnz       1122             15990       6 (=byte)
 *
 *    The ByteposData will be to the start of the ACTUAL (numerical) data,
 *    not the keyword/tag start (differs from Eclipse SCAN result here!)
 *
 * ARGUMENTS:
 *    fc              i     Filehandle (stream) to read from
 *    swap            o     SWAP status, 0 of False, 1 if True
 *    tags            o     A long *char where the tags are separated by a |
 *    rectypes        o     An array with record types: 1 = INT, 2 = FLOAT,
 *                          3 = DOUBLE, 4 = CHAR(STRING), 5 = BOOL, 6 = BYTE
 *    reclengths      o     An array with record lengths (no of elements)
 *    recstarts       o     An array with record starts (in bytes)
 *    maxkw           i     Max number of tags possible to read
 *    debug           i     Debug level
 *
 * RETURNS:
 *    Function: Number of keywords read. If problems, a negative value
 *    Resulting vectors will be updated.
 *
 *
 * NOTE:
 *    The ROFF format was developed independent of RMS, so integer varables
 *    in ROFF does not match integer grid parameters in RMS fully.  ROFF
 *    uses a signed int (4 byte). As integer values in RMS are always
 *    unsigned (non-negative) information will be lost if you try to import
 *    negative integer values from ROFF into RMS."
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ******************************************************************************
 */

/* ######################################################################### */
/* LOCAL FUNCTIONS                                                           */
/* ######################################################################### */

# define ROFFSTRLEN 100
# define ROFFARRLEN 15
# define TAGRECORDMAX 100
# define TAGDATAMAX 100

int _roffbinstring(FILE *fc, char *mystring)

{
    /* read a string; return the number of bytes (including 0 termination) */
    int i, ier;
    char mybyte;

    strcpy(mystring, "");

    for (i = 0;i < ROFFSTRLEN; i++) {
        ier = fread(&mybyte,1,1,fc);
	mystring[i] = mybyte;
        if (mybyte == '\0') return i + 1;
    }

    return -1;
}


int _scan_roff_bin_record(FILE *fc, int *swap, char tagname[ROFFSTRLEN],
                          long npos1, long *npos2, int *numrec,
                          char cname[ROFFARRLEN][ROFFSTRLEN],
                          char pname[ROFFARRLEN][ROFFSTRLEN],
                          int cntype[ROFFARRLEN], long bytepos[ROFFARRLEN],
                          long reclen[ROFFARRLEN], int debug)
{
    /*
     * tagname: is the name of the tag
     * npos1: is the byte INPUT position in the file
     * npos2: is the byte OUTPUT position, i.e. ready for next tag
     * cname: is the name of the subtag, as "array"
     * cntype: is data type: 1=int, 2=float, 3=double, 4=char, 5=byte
     * rnlen: is the record length, if > 1 then it is an array type.
     *        => if 1, then it may have several sub keys
     */

    char s[24] = "_scan_roff_bin_record";

    /* int swap = 0; */
    int ndat, nrec, i, n, ic;
    int bsize = 0;
    const int FAIL = -88;
    char tmpname[ROFFSTRLEN] = "";
    long ncum = 0;

    char cdum[ROFFSTRLEN] = "";
    int idum;
    float fdum;
    double ddum;
    unsigned char bdum;

    xtgverbose(debug);

    if (fseek(fc, npos1, SEEK_SET) != 0) return FAIL;

    ncum = ncum + npos1;

    nrec = 0;  /* record counter (subtag) */

    strcpy(tagname, "");

    for (i = 0; i < TAGRECORDMAX; i++) {

        ncum += _roffbinstring(fc, tmpname);

        xtg_speak(s, 2, "TMPNAME is %s, ncum = %ld", tmpname, ncum);

        if (npos1 == 0 && i == 0 &&
            strncmp(tmpname, "roff-bin", 8) != 0) {
            /* not a ROFF binary file! */
            return -9;
        }

        if (strncmp(tmpname, "tag", 3) == 0) {
            ncum += _roffbinstring(fc, tagname);

            if (strncmp(tagname, "eof", 3) == 0) {
                return 10;
            }

            /* now the rest of the record may contain of multiple e.g.: */
            /* float xoffset   4.61860625E+05 or */
            /* array float data 15990 */
            /* ... until */
            /* endtag */
            for (n = 0; n < TAGDATAMAX; n++) {
                ncum += _roffbinstring(fc, tmpname);

                if (strncmp(tmpname, "endtag", 6) == 0) {
                    *npos2 = ncum;
                    *numrec = nrec;
                    return 0;
                }

                strcpy(pname[nrec], "NAxxx");

                if (strncmp(tmpname, "int", 3) == 0) {
                    ncum += _roffbinstring(fc, cname[nrec]);
                    bytepos[nrec] = ncum;
                    ncum += fread(&idum, sizeof(int), 1, fc) * sizeof(int);

                    /* special treatment of byteswap */
                    if (strncmp(cname[nrec], "byteswaptest", 13) == 0) {
                        if (idum == 1) *swap = 0;
                        if (idum != 1) *swap = 1;
                    }

                    reclen[nrec] = 1;
                    cntype[nrec] = 1;
                    nrec++;
                }
                else if (strncmp(tmpname, "float", 5) == 0) {
                    ncum += _roffbinstring(fc, cname[nrec]);
                    bytepos[nrec] = ncum;
                    ncum += fread(&fdum, sizeof(float), 1, fc) * sizeof(float);
                    cntype[nrec] = 2;
                    reclen[nrec] = 1;
                    nrec++;

                }
                else if (strncmp(tmpname, "double", 6) == 0) {
                    /* never in use? */
                    ncum += _roffbinstring(fc, cname[nrec]);
                    bytepos[nrec] = ncum;
                    ncum += fread(&ddum, sizeof(double), 1, fc)
                        * sizeof(double);
                    cntype[nrec] = 3;
                    reclen[nrec] = 1;
                    nrec++;
                }
                else if (strncmp(tmpname, "char", 4) == 0) {
                    ncum += _roffbinstring(fc, cname[nrec]);
                    bytepos[nrec] = ncum;
                    /* char in ROFF is actually a string: */
                    ncum += _roffbinstring(fc, cdum);
                    cntype[nrec] = 4;
                    reclen[nrec] = 1;

                    /* special treatment of parameter names (extra info) */
                    if (strncmp(cname[nrec], "name", 4) == 0) {
                        if (strlen(cdum) == 0) strcpy(cdum, "unknown");
                        strcpy(pname[nrec], cdum);
                    }
                    nrec++;
                }
                else if (strncmp(tmpname, "bool", 4) == 0) {
                    ncum += _roffbinstring(fc, cname[nrec]);
                    bytepos[nrec] = ncum;
                    ncum += fread(&bdum, sizeof(unsigned char), 1, fc)
                        * sizeof(unsigned char);
                    cntype[nrec] = 5;
                    reclen[nrec] = 1;
                    nrec++;
                }
                else if (strncmp(tmpname, "byte", 4) == 0) {
                    ncum += _roffbinstring(fc, cname[nrec]);
                    bytepos[nrec] = ncum;
                    ncum += fread(&bdum, sizeof(unsigned char), 1, fc)
                        * sizeof(unsigned char);
                    cntype[nrec] = 6;
                    reclen[nrec] = 1;
                    nrec++;
                }
                else if (strncmp(tmpname, "array", 5) == 0) {
                    ncum += _roffbinstring(fc, tmpname);

                    if (strncmp(tmpname, "int", 3) == 0) {
                        bsize = 4;
                        ncum += _roffbinstring(fc, cname[nrec]);
                        ncum += fread(&ndat, sizeof(int), 1, fc) * sizeof(int);
                        if (*swap) SWAP_INT(ndat);
                        cntype[nrec] = 1;
                        bytepos[nrec] = ncum;
                        reclen[nrec] = ndat;
                        nrec++;
                    }
                    else if (strncmp(tmpname, "float", 5) == 0) {
                        bsize = 4;
                        ncum += _roffbinstring(fc, cname[nrec]);
                        ncum += fread(&ndat, sizeof(int), 1, fc) * sizeof(int);
                        if (*swap) SWAP_INT(ndat);
                        bytepos[nrec] = ncum;
                        cntype[nrec] = 2;
                        reclen[nrec] = ndat;
                        nrec++;

                    }

                    /* double never in use? */

                    else if (strncmp(tmpname, "char", 4) == 0) {
                    /* Note: arrays of type char (ie strings) have UNKNOWN */
                    /* lenghts; hence need special processing! -> bsize 0 */
                        bsize = 0;
                        ncum += _roffbinstring(fc, cname[nrec]);
                        ncum += fread(&ndat, sizeof(int), 1, fc) * sizeof(int);
                        if (*swap) SWAP_INT(ndat);
                        cntype[nrec] = 4;
                        bytepos[nrec] = ncum;
                        reclen[nrec] = ndat;
                        nrec++;
                    }
                    else if (strncmp(tmpname, "bool", 4) == 0) {
                        bsize = 1;
                        ncum += _roffbinstring(fc, cname[nrec]);
                        ncum += fread(&ndat, sizeof(int), 1, fc) * sizeof(int);
                        if (*swap) SWAP_INT(ndat);
                        bytepos[nrec] = ncum;
                        cntype[nrec] = 5;
                        reclen[nrec] = ndat;
                        nrec++;
                    }
                    else if (strncmp(tmpname, "byte", 4) == 0) {
                        bsize = 1;
                        ncum += _roffbinstring(fc, cname[nrec]);
                        ncum += fread(&ndat, sizeof(int), 1, fc) * sizeof(int);
                        if (*swap) SWAP_INT(ndat);
                        bytepos[nrec] = ncum;
                        cntype[nrec] = 6;
                        reclen[nrec] = ndat;
                        nrec++;
                    }

                    if (bsize == 0) {
                        for (ic = 0; ic < ndat; ic++) {
                            ncum += _roffbinstring(fc, cname[nrec]);
                        }
                    }
                    else{
                        ncum += bsize * ndat;
                        if (fseek(fc, ncum, SEEK_SET) != 0) return FAIL;
                    }
                }
            }
        }
    }

    return EXIT_SUCCESS;
}


/* ######################################################################### */
/* LIBRARY FUNCTION                                                          */
/* ######################################################################### */

long grd3d_scan_roffbinary (FILE *fc, int *swap, char *tags, int *rectypes,
                            long *reclengths, long *recstarts, long maxkw,
                            int debug)
{

    char s[24] = "grd3d_scan_roffbinary";
    char tagname[ROFFSTRLEN] = "";
    char cname[ROFFARRLEN][ROFFSTRLEN];
    char pname[ROFFARRLEN][ROFFSTRLEN];
    int i, j, numrec, ios, cntype[ROFFARRLEN];
    long npos1, npos2, bytepos[ROFFARRLEN], reclen[ROFFARRLEN];
    long nrec=0;

    xtgverbose(debug);

    xtg_speak(s, 2, "Scanning ROFF ...");

    npos1 = 0;

    ios = 0;

    tags[0] = '\0';

    rewind(fc);

    for (i = 0; i < maxkw; i++) {
        tagname[0] = '\0';
        ios = _scan_roff_bin_record(fc, swap, tagname, npos1, &npos2, &numrec,
                                    cname, pname, cntype,
                                    bytepos, reclen, debug);

        if (ios == -9) {
            xtg_error(s, "Not a ROFF binary file. STOP!");
            return ios;
        }
        else if (ios < 0) {
            xtg_error(s, "Unspesified error when reading ROFF binary: %d",
                      ios);
            return -10;
        }

        if (strcmp(tagname, "eof") == 0 || ios == 10) break;

        for (j = 0; j < numrec; j++) {
            xtg_speak(s, 2, "Tag is <%s>, subtags: <%s>, "
                      "bytepos: <%ld>, reclen: <%ld>, "
                      "npos1 and npos2: <%ld> <%ld>",
                      tagname, cname[j], bytepos[j], reclen[j], npos1, npos2);
            strcat(tags, tagname);
            strcat(tags, "!");
            strcat(tags, cname[j]);

            /* add a third item if parameter name */
            if (strncmp(cname[j], "name", 4) == 0 &&
                strncmp(pname[j], "NAxxx", 2) != 0) {

                strcat(tags, "!");
                strcat(tags, pname[j]);
            }
            strcat(tags, "|");
            rectypes[nrec] = cntype[j];
            reclengths[nrec] = reclen[j];
            recstarts[nrec] = bytepos[j];
            nrec ++;
        }

        npos1 = npos2;
    }
    return nrec;
}
