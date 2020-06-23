/*
 ***************************************************************************************
 *
 * NAME:
 *    grdcp3d_imp_roff_zcornsv.c
 *
 * DESCRIPTION:
 *    Fast import of ZCORNSV data in xtgformat=2. This format is C ordered in rows
 *    and columns.
 *
 * ARGUMENTS:
 *    fc                 i     Filehandle (stream) to read from
 *    swap               i     SWAP status, 0 if False, 1 if True
 *    bytepos            i     The byte position to the code data
 *    nncol, .., nnlay   i     Number of cell nodes in 3D (ncol + 1, etc)
 *    xoffset, ...       i     Offsets and scaling from Roxar API
 *    zcornsv           i/o    Pointer array to be updated (initilised outside)
 *    nitems             i     Length of coordsv array (nncol * nncol * 6)
 *
 * RETURNS:
 *    EXIT_STATUS (0) if OK, updated float pointer *coordsv
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
grdcp3d_imp_roffbin_zcornsv(FILE *fc,
                            int swap,
                            long bytepos,
                            int nncol,
                            int nnrow,
                            int nnlay,
                            float xoffset,
                            float yoffset,
                            float zoffset,
                            float xscale,
                            float yscale,
                            float zscale,
                            int *splitenz,
                            long nsplitenz,
                            float *zcornsv,
                            long nitems)
{
    /* Imports a ROFF binary array, update pointer */

    float afloat;
    float **pillar = x_allocate_2d_float(nnlay, 4);

    logger_info(LI, FI, FU, "Reading ZCORNS...");
    logger_info(LI, FI, FU, "Reading from byte position %ld with swap %d", bytepos,
                swap);

    fseek(fc, bytepos, SEEK_SET);

    if (swap == 0) {

        // just a global swap to read without swap eval for every item to improve speed
        long ic = 0;
        int i, j;

        for (i = 0; i < nncol; i++) {
            for (j = 0; j < nnrow; j++) {
                int k;
                for (k = 0; k < nnlay; k++) {

                    int nsplit = splitenz[i * nnrow * nnlay + j * nnlay + k];

                    int n;
                    if (nsplit == 4) {
                        for (n = 0; n < 4; n++) {
                            if (fread(&afloat, 4, 1, fc) != 1)
                                return EXIT_FAILURE;
                            pillar[k][n] = (afloat + zoffset) * zscale;
                        }
                    } else if (nsplit == 1) {
                        if (fread(&afloat, 4, 1, fc) != 1)
                            return EXIT_FAILURE;
                        for (n = 0; n < 4; n++) {
                            pillar[k][n] = (afloat + zoffset) * zscale;
                        }
                    } else {
                        logger_critical(
                          LI, FI, FU, "Probably a bug in %s, nsplit is %d for %d %d %d",
                          FU, nsplit, i, j, k);
                        exit(-989);
                    }
                }
                for (k = (nnlay - 1); k >= 0; k--) {
                    int n;
                    for (n = 0; n < 4; n++) {
                        zcornsv[ic++] = pillar[k][n];
                    }
                }
            }
        }
    } else {

        long ic = 0;
        size_t i, j;

        for (i = 0; i < nncol; i++) {
            for (j = 0; j < nnrow; j++) {
                int k;
                for (k = 0; k < nnlay; k++) {

                    int nsplit = splitenz[(i * nnrow + j) * nnrow + k];

                    int n;
                    if (nsplit == 4) {
                        for (n = 0; n < 4; n++) {
                            if (fread(&afloat, 4, 1, fc) != 1)
                                return EXIT_FAILURE;
                            SWAP_FLOAT(afloat);
                            pillar[k][n] = (afloat + zoffset) * zscale;
                        }
                    } else if (nsplit == 1) {
                        if (fread(&afloat, 4, 1, fc) != 1)
                            return EXIT_FAILURE;
                        SWAP_FLOAT(afloat);
                        for (n = 0; n < 4; n++) {
                            pillar[k][n] = (afloat + zoffset) * zscale;
                        }
                    } else {
                        logger_critical(LI, FI, FU, "Probably a bug in %s", FU);
                        exit(-989);
                    }
                }
                for (k = (nnlay - 1); k >= 0; k--) {
                    int n;
                    for (n = 0; n < 4; n++) {
                        zcornsv[ic++] = pillar[k][n];
                    }
                }
            }
        }
    }

    logger_info(LI, FI, FU, "Reading ZCORNSV done");

    x_free_2d_float(pillar);

    return EXIT_SUCCESS;
}
