/*
 ***************************************************************************************
 *
 * NAME:
 *    surf_import_petromod_bin.c
 *
 *(S):
 *
 *
 * DESCRIPTION:
 *    Imports a surface map on Petromod pdm binary format.
 *
 * ARGUMENTS:
 *    fhandle        i     Filehandle (steered from caller)
 *    mode           i     0 = scan mode to find mx, my, etc; 1 = normal mode
 *    undef          i     Undef value in file (evaluated from dsc after scan)
 *    dsc            o     The description field (will be parsed in Python)
 *    mx             i     Map dimension X (I) pointer
 *    my             i     Map dimension Y (J) pointer
 *    surfzv         o     1D pointer to map/surface values pointer array
 *    nsurf          i     No. of map nodes (for allocation from Python/SWIG)
 *
 * RETURNS:
 *    Void, pointer arrays are updated

 * TODO/ISSUES/BUGS/NOTES:
 *
 * LICENCE:
 *    See XTGeo licence
 ***************************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"
#include "logger.h"
#include <math.h>

void
surf_import_petromod_bin(FILE *fc,
                         int mode,
                         float undef,
                         char *dsc,
                         int mx,
                         int my,
                         double *surfzv,
                         long nsurf)
{

    logger_info(LI, FI, FU, "Read PETROMOD binary map file: %s", FU);

    if (mx * my != nsurf)
        logger_critical(LI, FI, FU, "mx * my != nsurf, bug in %s", FU);

    if (mode == 0)
        logger_info(LI, FI, FU, "Scan mode!");
    if (mode == 1)
        logger_info(LI, FI, FU, "Values mode!");

    fseek(fc, 0, SEEK_SET);

    /* check endianess */
    int swap = 0;
    if (x_swap_check() == 1)
        swap = 1;

    float myfloat;
    if (fread(&myfloat, 4, 1, fc) != 1)
        logger_critical(LI, FI, FU, "Error in fread() in %s", FU);

    if (swap)
        SWAP_FLOAT(myfloat);
    logger_info(LI, FI, FU, "TAG %f", myfloat);

    int ier = fscanf(fc, "%300s", dsc);
    logger_info(LI, FI, FU, "IER from fscanf() is %d in %s", ier, FU);

    logger_info(LI, FI, FU, "TAG %s", dsc);

    if (mode == 0)
        return; /* scan mode */

    int nlen = strnlen(dsc, 500);
    logger_info(LI, FI, FU, "Length of description is %d", nlen);

    fseek(fc, nlen + 5, SEEK_SET);

    int in, jn;
    long ic = 0;
    for (in = 0; in < mx; in++) {
        for (jn = 0; jn < my; jn++) {
            if (fread(&myfloat, 4, 1, fc) != 1)
                logger_critical(LI, FI, FU, "Error in fread() in %s", FU);
            if (swap)
                SWAP_FLOAT(myfloat);
            if (fabs(myfloat - undef) < FLOATEPS)
                myfloat = UNDEF;
            surfzv[ic++] = (double)myfloat;
        }
    }

    logger_info(LI, FI, FU, "Importing Petromod binary from file done");
}
