/*
 ***************************************************************************************
 *
 * NAME:
 *    grdcp3d_imp_roff_coordsv.c
 *
 * DESCRIPTION:
 *    Fast import of COORDSV data in xtgformat=2. This format is C ordered in rows
 *    and columns
 *
 * ARGUMENTS:
 *    fc              i     Filehandle (stream) to read from
 *    swap            i     SWAP status, 0 if False, 1 if True
 *    bytepos         i     The byte position to the code data
 *    nncol, nnrow    i     Number of nodes (ncol + 1, nrow + 1)
 *    xoffset, ...    i     Offsets and scaling from Roxar API
 *    coordsv        i/o    Pointer array to be updated (initilised outside)
 *    nitems          i     Length of coordsv array (nncol * nncol * 6)
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
grdcp3d_imp_roffbin_coordsv(FILE *fc,
                            int swap,
                            long bytepos,
                            int nncol,
                            int nnrow,
                            float xoffset,
                            float yoffset,
                            float zoffset,
                            float xscale,
                            float yscale,
                            float zscale,
                            double *coordsv,
                            long nitems)
{
    /* Imports a ROFF binary array, update pointer */

    float afloat;
    double top[3] = { 0.0 }, base[3] = { 0.0 };
    size_t i, j, n;

    logger_info(LI, FI, FU, "Reading COORDSV from byte position %ld with swap %d",
                bytepos, swap);

    fseek(fc, bytepos, SEEK_SET);

    if (swap == 0) {
        // just a global swap to read without swap eval for every item to improve speed
        for (i = 0; i < nncol; i++) {
            for (j = 0; j < nnrow; j++) {
                for (n = 0; n < 6; n++) {
                    if (fread(&afloat, 4, 1, fc) != 1)
                        return EXIT_FAILURE;
                    if (n == 0) {
                        base[0] = (afloat + xoffset) * xscale;
                    } else if (n == 1) {
                        base[1] = (afloat + yoffset) * yscale;
                    } else if (n == 2) {
                        base[2] = (afloat + zoffset) * zscale;
                    } else if (n == 3) {
                        top[0] = (afloat + xoffset) * xscale;
                    } else if (n == 4) {
                        top[1] = (afloat + yoffset) * yscale;
                    } else if (n == 5) {
                        top[2] = (afloat + zoffset) * zscale;
                    }
                }
                for (n = 0; n < 3; n++)
                    coordsv[(i * nnrow + j) * 6 + n] = top[n];
                for (n = 3; n < 6; n++)
                    coordsv[(i * nnrow + j) * 6 + n] = base[n - 3];
            }
        }
    } else {
        for (i = 0; i < nncol; i++) {
            for (j = 0; j < nnrow; j++) {
                for (n = 0; n < 6; n++) {
                    if (fread(&afloat, 4, 1, fc) != 1)
                        return EXIT_FAILURE;

                    SWAP_FLOAT(afloat);

                    if (n == 0) {
                        base[0] = (afloat + xoffset) * xscale;
                    } else if (n == 1) {
                        base[1] = (afloat + yoffset) * yscale;
                    } else if (n == 2) {
                        base[2] = (afloat + zoffset) * zscale;
                    } else if (n == 3) {
                        top[0] = (afloat + xoffset) * xscale;
                    } else if (n == 4) {
                        top[1] = (afloat + yoffset) * yscale;
                    } else if (n == 5) {
                        top[2] = (afloat + zoffset) * zscale;
                    }
                }
                for (n = 0; n < 3; n++)
                    coordsv[(i * nnrow + j) * 6 + n] = top[n];
                for (n = 3; n < 6; n++)
                    coordsv[(i * nnrow + j) * 6 + n] = base[n - 3];
            }
        }
    }

    logger_info(LI, FI, FU, "Reading COORDSV done");

    return EXIT_SUCCESS;
}
