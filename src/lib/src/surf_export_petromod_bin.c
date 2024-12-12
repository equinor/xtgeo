/*
 ***************************************************************************************
 *
 * NAME:
 *    surf_export_petromod_bin.c
 *
 * DESCRIPTION:
 *    Export a map to Petromod binary. Preprocessing avd text dec field done in Python
 *
 * ARGUMENTS:
 *    fc             i     File handle
 *    dsc            i     Description field
 *    surfzv         i     surface vector
 *    nsurf          i     Number of elements
 *
 * RETURNS:
 *    Void
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ***************************************************************************************
 */
#include <stdio.h>
#include <xtgeo/xtgeo.h>
#include "common.h"
#include "logger.h"

void
surf_export_petromod_bin(FILE *fc, char *dsc, double *surfzv, long nsurf)
{

    logger_info(LI, FI, FU, "Write Petromod binary map file... (%s)", FU);

    int swap = x_swap_check();

    if (fc == NULL) {
        throw_exception("Cannot open file in: surf_export_petromod_bin");
        return;
    }
    int someid = 587405668;
    if (swap == 1)
        SWAP_INT(someid);

    fwrite(&someid, 4, 1, fc);

    fprintf(fc, "%s", dsc);

    char mynull[1] = { '\0' };
    fwrite(&mynull, 1, 1, fc);

    float myvalue;
    int ic;
    for (ic = 0; ic < nsurf; ic++) {
        myvalue = (float)surfzv[ic];

        if (swap == 1)
            SWAP_FLOAT(myvalue);

        if (fwrite(&myvalue, 4, 1, fc) != 1) {
            throw_exception(
              "Error writing to Storm format. Bug in: surf_export_petromod_bin");
            return;
        }
    }
    logger_info(LI, FI, FU, "Write Petromod binary map file... done");
}
