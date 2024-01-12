/*
 ***************************************************************************************
 *
 * NAME:
 *    x_verify_vectorlengths.c
 *
 * DESCRIPTION:
 *    Verify typical grid vector lengths
 *
 * ARGUMENTS:
 *    ncol, nrow, nlay  i     Dimensions
 *    ncoord            i     Lenghts of coord vector (use 0 or less to skip)
 *    nzcorn            i     Lenghts of zcorn vector (use 0 or less to skip)
 *    ntot              i     Array: Lenghts of ntot vector (use 0 or less to skip)
 *    ntotlen           i     Length of ntot array
 *    format            i     XTG format 1 or 2
 *
 * RETURNS:
 *    Status, EXIT_FAILURE or EXIT_SUCCESS
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    CF XTGeo's LICENSE
 ***************************************************************************************
 */
#include <stdlib.h>

#include <xtgeo/xtgeo.h>

#include "common.h"
#include "logger.h"

int
x_verify_vectorlengths(long ncol,
                       long nrow,
                       long nlay,
                       long ncoord,
                       long nzcorn,
                       long *ntot,
                       int ntotlen,
                       int format)
{
    long ncoordtrue = (ncol + 1) * (nrow + 1) * 6;
    long ntottrue = ncol * nrow * nlay;
    long nzcorntrue = (ncol + 1) * (nrow + 1) * (nlay + 1) * 4;
    /* Default to XTG format 2. */
    if (format == XTGFORMAT1) {
        nzcorntrue = ncol * nrow * (nlay + 1) * 4;
    }

    if (ncoord > 0 && (ncoord != ncoordtrue)) {
        throw_exception("Error in ncoord check: ncoord > 0 and ncoord != ncoordtrue");
        return EXIT_FAILURE;
    }
    if (nzcorn > 0 && (nzcorn != nzcorntrue)) {
        throw_exception("Error in ncoord check: nzcorn > 0 and nzcorn != nzcoordtrue");
        return EXIT_FAILURE;
    }
    for (int i = 0; i < ntotlen; i++) {
        if (ntot[i] > 0 && (ntot[i] != ntottrue)) {
            logger_error(LI, FI, FU, "Error in ntot check %d: %ld vs %ld (true)", i,
                         ntot[i], ntottrue);
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
