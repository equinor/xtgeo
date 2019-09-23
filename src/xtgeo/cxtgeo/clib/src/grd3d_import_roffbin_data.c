/*
 *******************************************************************************
 *
 * Read ROFF binary data (new 2018 import model)
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
 *    grd3d_imp_roffbin_data.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Note, this is for SINGLE data. For arrays, see grd3d_imp_roffbin_arr.c
 *
 *    This is a new line of ROFF handling function (from 2018). A seperate
 *    scan ROFF binary must be done first, which will return e.g.:
 *
 *    NameEntry              ByteposData      LenData     Datatype
 *    scale!xscale           94               1           2 (=float)
 *    zvalues!splitEnz       1122             15990       6 (=byte)
 *
 *    The ByteposData will be to the start of the ACTUAL (numerical) data,
 *    not the keyword/tag start (differs from Eclipse SCAN result here!)
 *
 *    This reoutine will the actual data. This reads single data items.
 *
 * ARGUMENTS:
 *    fc              i     Filehandle (stream) to read from
 *    swap            i     SWAP status, 0 of False, 1 if True
 *    rectypes        i     An integer for record types: 1 = INT, 2 = FLOAT,
 *                          3 = DOUBLE, 4 = CHAR(STRING), 5 = BOOL, 6 = BYTE
 *    recstarts       i     A long int with record starts (in bytes)
 *    p_int           o     pointer to return data, if int
 *    p_dbl           o     pointer to return data, if double
 *    debug           i     Debug level
 *
 * RETURNS:
 *    0 if OK
 *    Resulting pointers will be updated.
 *
 * NOTE:
 *    "The ROFF format was developed independent of RMS, so integer varables
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
/* LIBRARY FUNCTION                                                          */
/* ######################################################################### */

int grd3d_imp_roffbin_data (FILE *fc, int swap, int dtype,
                            long bytepos, int *p_int,
                            float *p_flt, int debug)
{

    char s[24] = "grd3d_imp_roffbin_data";
    int anint, iok;
    float afloat;

    xtgverbose(debug);

    xtg_speak(s, 2, "Running %s", s);

    fseek(fc, bytepos, SEEK_SET);

    if (dtype == 1) {
        iok = fread(&anint, 4, 1, fc);
        if (swap==1) SWAP_INT(anint);
        *p_int = anint;
    }

    else if (dtype == 2) {
        iok = fread(&afloat, 4, 1, fc);
        if (swap==1) SWAP_FLOAT(afloat);
        *p_flt = afloat;
    }

    return EXIT_SUCCESS;
}
