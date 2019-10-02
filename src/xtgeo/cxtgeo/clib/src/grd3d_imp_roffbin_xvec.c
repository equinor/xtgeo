/*
 ***************************************************************************************
 *
 * Read ROFF property data (after a scanning is done first)
 *
 ***************************************************************************************
 */

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "libxtg.h"
#include "libxtg_.h"

/*
 ***************************************************************************************
 *
 * NAME:
 *    grd3d_imp_roffbin_xvec.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Get the pointer to the a general X (float, int, ...) array
 *
 * ARGUMENTS:
 *    fc              i     Filehandle (stream) to read from
 *    swap            i     SWAP status, 0 of False, 1 if True
 *    bytepos         i     The byte position to the code data
 *    xvec           i/o    Pointer to floatInt/etc data
 *    n*vec           i     Length of float array
 *
 * RETURNS:
 *    Updated float pointer *vec
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ***************************************************************************************
 */

int grd3d_imp_roffbin_fvec(FILE *fc, int swap, long bytepos, float *fvec, long nfvec)
{
    /* Imports a ROFF binary array, update pointer */

    float afloat;
    long i;

    fseek(fc, bytepos, SEEK_SET);

    for (i = 0; i < nfvec; i++) {
        if (fread(&afloat, 4, 1, fc) != 1) exit(EXIT_FAILURE);
        if (swap==1) SWAP_FLOAT(afloat);
        if (afloat == -999.0) afloat = UNDEF;
        fvec[i] = afloat;
    }

    return EXIT_SUCCESS;
}

int grd3d_imp_roffbin_ivec(FILE *fc, int swap, long bytepos, int *ivec, long nivec)
{
    /* Imports a ROFF binary array, update pointer */

    int anint;
    long i;

    fseek(fc, bytepos, SEEK_SET);

    for (i = 0; i < nivec; i++) {
        if (fread(&anint, 4, 1, fc) != 1) exit(EXIT_FAILURE);
        if (swap==1) SWAP_FLOAT(anint);
        if (anint == -999.0) anint = UNDEF_INT;
        ivec[i] = anint;
    }

    return EXIT_SUCCESS;
}

int grd3d_imp_roffbin_bvec(FILE *fc, int swap, long bytepos, int *bvec, long nbvec)
{
    /* Imports a ROFF binary array if type , update pointer. NB convert to INT! */

    unsigned char achar;
    long i;
    int anint;

    fseek(fc, bytepos, SEEK_SET);

    for (i = 0; i < nbvec; i++) {
        if (fread(&achar, 1, 1, fc) != 1) exit(EXIT_FAILURE);
        anint = (int)achar;
        if (anint == 255) anint = UNDEF_INT;
        bvec[i] = anint;
    }

    return EXIT_SUCCESS;
}
