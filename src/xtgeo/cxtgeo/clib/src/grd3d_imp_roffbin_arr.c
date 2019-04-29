/*
 ******************************************************************************
 *
 * Read ROFF proerty data (after a scanning is done first)
 *
 ******************************************************************************
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
 *    grd3d_imp_roffbin_arr.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    This routine goes directly to the byte position(s) which are found from
 *    scanning, and reads the array. Map into XTGeo format.
 *
 * ARGUMENTS:
 *    fc              i     Filehandle (stream) to read from
 *    swap            i     SWAP status, 0 of False, 1 if True
 *    nx, ny, nz      i     Dimensions
 *    bytepos         i     The byte position to the code data
 *    dtype           i     Data type to read: 1=int, 2=float, 3=double,
 *                          4=char, 5=byte
 *    farray         i/o    Pointer 1D to float array (must be allocated pre)
 *    nfarray         i     Length of float array
 *    iarray         i/o    Pointer 1D to int array (must be allocated pre)
 *    niarray         i     Length of int array
 *    debug           i     Debug level
 *
 * RETURNS:
 *    Function: 0 if success. Updated pointers.
 *
 * NOTES:
 *    Both byte and bool data will be stored as 32 bit ints in XTGeo.
 *
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

int grd3d_imp_roffbin_arr (FILE *fc, int swap, int nx, int ny, int nz,
                           long bytepos, int dtype, float *farray,
                           long nfarray, int *iarray, long niarray, int debug)
{
    /* Imports a ROFF binary array which has nx * ny * nz data points */

    char s[24] = "grd3d_imp_roffbin_arr";
    int anint;
    float afloat;
    double adouble;
    char abyte;

    int i, j, k, iok, kactual;
    long ipos;

    xtgverbose(debug);


    xtg_speak(s, 2, "Importing a roff array with NX * NY * NZ entries");

    fseek(fc, bytepos, SEEK_SET);

    ipos = -1;
    for (i = 1; i <= nx; i++) {
        for (j = 1; j <= ny; j++) {
            for (k = nz; k >= 1; k--) {

                kactual = nz - k + 1;

                ipos = x_ijk2ic(i, j, k, nx, ny, nz, 0);

                if (dtype == 1) {
                    iok = fread(&anint, 4, 1, fc);
                    if (swap==1) SWAP_INT(anint);
                    if (anint == -999) anint = UNDEF_INT;
                    iarray[ipos] = anint;
                }

                else if (dtype == 2) {
                    iok = fread(&afloat, 4, 1, fc);
                    if (swap==1) SWAP_FLOAT(afloat);
                    if (afloat == -999.0) afloat = UNDEF;
                    farray[ipos] = afloat;
                }

                else if (dtype == 3) {
                    /* not sure if this ever happens; no double in ROFF? */
                    iok = fread(&adouble, 8, 1, fc);
                    if (swap==1) SWAP_DOUBLE(adouble);
                    if (adouble == -999.0) adouble = UNDEF;
                    farray[ipos] = (float)adouble;
                }

                else if (dtype == 5) {
                    iok = fread(&abyte, 1, 1, fc);
                    anint = abyte;  /* convert to int */
                    if (anint == 255) anint = UNDEF_INT;
                    iarray[ipos] = anint;
                }
            }
        }
    }
    return EXIT_SUCCESS;
}
