/*
 ***************************************************************************************
 *
 * Read ROFF property data (after a scanning is done first)
 *
 ***************************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 ***************************************************************************************
 *
 * NAME:
 *    grdcp3d_imp_roffbin_prop_xvec.c
 *
 * DESCRIPTION:
 *    Read from ROFF and maps pointer directlry to XTGeo layout, which is in
 *    xtgformat=2 C ordered but with K starting at top, not at base as in ROFF.
 *
 * ARGUMENTS:
 *    fc              i     Filehandle (stream) to read from
 *    swap            i     SWAP status, 0 of False, 1 if True
 *    bytepos         i     The byte position to the code data
 *    ncol .. nlay    i     Dimensions
 *    pvec           i/o    Pointer to float/int/etc array property data
 *    nvec            i     Length of array (for SWIG)
 *
 * RETURNS:
 *    Success or failure to be handled at client. Updated pointer array *vec
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ***************************************************************************************
 */

int
grdcp3d_imp_roffbin_prop_ivec(FILE *fc,
                              int swap,
                              long bytepos,
                              long ncol,
                              long nrow,
                              long nlay,
                              int *pvec,
                              long nvec)
{
    /* Imports a ROFF binary array, update pointer */

    size_t anint;

    fseek(fc, bytepos, SEEK_SET);

    if (swap == 0) {
        long ic = 0;
        long i;
        for (i = 0; i < ncol; i++) {
            long j;
            for (j = 0; j < nrow; j++) {
                long k;
                for (k = (nlay - 1); k >= 0; k--) {
                    if (fread(&anint, 4, 1, fc) != 1)
                        return EXIT_FAILURE;
                    if (anint == -999.0)
                        anint = UNDEF_INT;
                    pvec[ic++] = anint;
                }
            }
        }
    } else {
        long ic = 0;
        long i;
        for (i = 0; i < ncol; i++) {
            long j;
            for (j = 0; j < nrow; j++) {
                long k;
                for (k = (nlay - 1); k >= 0; k--) {
                    if (fread(&anint, 4, 1, fc) != 1)
                        return EXIT_FAILURE;
                    SWAP_INT(anint);
                    if (anint == -999.0)
                        anint = UNDEF_INT;
                    pvec[ic++] = anint;
                }
            }
        }
    }
    return EXIT_SUCCESS;
}

int
grdcp3d_imp_roffbin_prop_bvec(FILE *fc,
                              int swap,
                              long bytepos,
                              long ncol,
                              long nrow,
                              long nlay,
                              int *pvec,
                              long nvec)
{
    // a byte vector does not need swap, and is converted to int array!

    unsigned char abyte;
    int anint;

    fseek(fc, bytepos, SEEK_SET);

    long ic = 0;
    long i;
    for (i = 0; i < ncol; i++) {
        long j;
        for (j = 0; j < nrow; j++) {
            long k;
            for (k = (nlay - 1); k >= 0; k--) {
                if (fread(&abyte, 1, 1, fc) != 1)
                    return EXIT_FAILURE;
                anint = (int)abyte;
                if (anint == 255)
                    anint = UNDEF_INT;
                pvec[ic++] = anint;
            }
        }
    }
    return EXIT_SUCCESS;
}
