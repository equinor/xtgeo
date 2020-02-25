/*
****************************************************************************************
*
* NAME:
*    x_verify_vectorlengths.c
*
* DESCRIPTION:
*    Verify typical grid vector lengths
*
* ARGUMENTS:
*    nx, ny, nz     i     Dimensions
*    ncoord         i     Lenghts of coord vector (use 0 or less to skip)
*    nzcorn         i     Lenghts of zcorn vector (use 0 or less to skip)
*    ntot           i     Array: Lenghts of ntot vector (use 0 or less to skip)
*    ntotlen        i     Length of ntot array
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

#include "libxtg.h"
#include "libxtg_.h"
#include "logger.h"

int
x_verify_vectorlengths(int nx,
                       int ny,
                       int nz,
                       long ncoord,
                       long nzcorn,
                       long *ntot,
                       int ntotlen)
{

    long ncoordtrue = (nx + 1) * (ny + 1) * 6;
    long nzcorntrue = nx * ny * (nz + 1) * 4;
    long ntottrue = nx * ny * nz;

    if (ncoord > 0 && (ncoord != ncoordtrue)) {
        logger_error(LI, FI, FU, "Error in ncoord check: %ld vs %ld (true)", ncoord,
                     ncoordtrue);
        return EXIT_FAILURE;
    }
    if (nzcorn > 0 && (nzcorn != nzcorntrue)) {
        logger_error(LI, FI, FU, "Error in nzcorn check: %ld vs %ld (true)", nzcorn,
                     nzcorntrue);
        return EXIT_FAILURE;
    }
    int i;
    for (i = 0; i < ntotlen; i++) {
        if (ntot[i] > 0 && (ntot[i] != ntottrue)) {
            logger_error(LI, FI, FU, "Error in ntot check %d: %ld vs %ld (true)", i,
                         ntot[i], ntottrue);
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
