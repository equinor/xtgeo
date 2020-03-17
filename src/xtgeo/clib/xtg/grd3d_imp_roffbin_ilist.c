/*
 ***************************************************************************************
 *
 * NAME:
 *    grd3d_imp_roffbin_ilist.c
 *
 * DESCRIPTION:
 *    Read ROFF integer list like data (after a scanning is done first).
 *    This routine goes directly to the byte position(s) which are found from
 *    scanning, and reads the array.
 *
 * ARGUMENTS:
 *    fc              i     Filehandle (stream) to read from
 *    swap            i     SWAP status, 0 of False, 1 if True
 *    bytepos         i     The byte position to the code data
 *    iarray         i/o    Pointer 1D to int array (must be allocated pre)
 *    niarray         i     Length of int array
 *    debug           i     Debug level
 *
 * RETURNS:
 *    Function: 0 if success. Updated pointers.
 *
 * NOTES:
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
 ***************************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"
#include "logger.h"

int
grd3d_imp_roffbin_ilist(FILE *fc, int swap, long bytepos, int *iarray, long niarray)
{

    int anint, iok;

    long ipos;

    fseek(fc, bytepos, SEEK_SET);

    for (ipos = 0; ipos < niarray; ipos++) {
        iok = fread(&anint, 4, 1, fc);
        if (iok != 1)
            exit(EXIT_FAILURE);
        if (swap == 1)
            SWAP_INT(anint);
        if (anint == -999)
            anint = UNDEF_INT;
        iarray[ipos] = anint;
    }
    return EXIT_SUCCESS;
}
