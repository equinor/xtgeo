/*
****************************************************************************************
*
* NAME:
*    grdcp3d_export_xtgeo_grid.c
*
* DESCRIPTION:
*    Export to native binary xtgeo format. This format is quite compact but it also
*    have the possibility to add metadata as a json structure in the end.
*    The position of the metadata will be at 4byte + 16byte + 3*8byte + ncoord*8byte +
*    nzcorn * 4byte + nact * 4byte + 16byte + 1?
*
*    Stuff like subgridinfo will be in the metadata
*
* ARGUMENTS:
*    ncol, nrow, nlay    i     NCOL, NROW, NLAY
*    zcornsv             i     ZCORN array w/ len
*    actnumsv            i     ACTNUM array w/ len
*    metadata            i     String (variable length) with metadata
*    filehandle          i     File handle
*
* RETURNS:
*    Void function
*
* LICENCE:
*    CF. XTGeo license
***************************************************************************************
*/

#include "libxtg.h"
#include "libxtg_.h"
#include "logger.h"

void
grdcp3d_export_xtgeo_grid(long ncol,
                          long nrow,
                          long nlay,
                          double *coordsv,
                          long ncoord,
                          float *zcornsv,
                          long nzcorn,
                          int *actnumsv,
                          long nact,
                          char *metadata,
                          FILE *fc)

{
    /*
     *----------------------------------------------------------------------------------
     * Initial part
     *----------------------------------------------------------------------------------
     */

    logger_info(LI, FI, FU, "Initial part...");

    char magicsta[16] = ">XTGGRIDCP3D_V1";  // 15 letters + one null -> 16
    char magicend[16] = "<XTGGRIDCP3D_V1";

    // for byteswap identification (future)
    int anint = 1;
    fwrite(&anint, 1, sizeof(int), fc);

    // 16 char/byte magic string
    fputs(magicsta, fc);

    // size of grid as longs (for potential handling of very big grids)
    fwrite(&ncol, sizeof(long), 1, fc);
    fwrite(&nrow, sizeof(long), 1, fc);
    fwrite(&nlay, sizeof(long), 1, fc);

    /*
     *----------------------------------------------------------------------------------
     * Corner lines
     *----------------------------------------------------------------------------------
     */

    logger_info(LI, FI, FU, "Corner lines...");

    fwrite(coordsv, sizeof(double), ncoord, fc);
    /*
     *----------------------------------------------------------------------------------
     * Z corner values
     *----------------------------------------------------------------------------------
     */

    logger_info(LI, FI, FU, "ZCORNS...");

    fwrite(zcornsv, sizeof(float), nzcorn, fc);

    /*
     *----------------------------------------------------------------------------------
     * ACTNUMS
     *----------------------------------------------------------------------------------
     */
    logger_info(LI, FI, FU, "ACTNUMS...");

    fwrite(actnumsv, sizeof(int), nact, fc);

    /*
     *----------------------------------------------------------------------------------
     * END magic + Metadata
     *----------------------------------------------------------------------------------
     */
    logger_info(LI, FI, FU, "End tag and Metadata...");

    fputs(magicend, fc);
    fprintf(fc, "%s", metadata);

    logger_info(LI, FI, FU, "Export done");
}
