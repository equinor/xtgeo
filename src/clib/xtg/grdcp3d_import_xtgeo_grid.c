/*
****************************************************************************************
*
* NAME:
*    grd3d_import_xtgeo_grid.c
*
* DESCRIPTION:
*    Import from native binary xtgeo format
*
* ARGUMENTS:
*    mode                i     0: read header, 1: read all, 2: read metadataonly
*    ncol, nrow, nlay    o     NCOL, NROW, NLAY
*    num_subgrds         o     Number of subgrids
*    coordsv             o     COORD array w/ len
*    zcornsv             o     ZCORN array w/ len
*    actnumsv            o     ACTNUM array w/ len
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

// max length of metadata; make sure that char *swig_bnd_char_100k in libxtg.h matches
#define METALEN 100000

int
grdcp3d_import_xtgeo_grid(int mode,
                          long *ncol,
                          long *nrow,
                          long *nlay,
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

    logger_info(LI, FI, FU, "Mode is %d", mode);

    logger_info(LI, FI, FU, "Initial part...");
    fseek(fc, 0, SEEK_SET);

    int anint;
    if (fread(&anint, sizeof(int), 1, fc) != 1) {
        return -100;
    }
    // future for possible byteswap detection
    if (anint != 1)
        return -101;

    // 15 char magic string
    char xtgeoid[16];
    fgets(xtgeoid, 16, fc);

    if (strncmp(xtgeoid, ">XTGGRIDCP3D_V1", 15) != 0)
        return -102;

    if (fread(ncol, sizeof(long), 1, fc) != 1)
        return -110;
    if (fread(nrow, sizeof(long), 1, fc) != 1)
        return -111;
    if (fread(nlay, sizeof(long), 1, fc) != 1)
        return -112;

    if (mode == 0) {
        logger_info(LI, FI, FU, "Initial part... done");
        return EXIT_SUCCESS;
    }

    long nncol = *ncol + 1;
    long nnrow = *nrow + 1;
    long nnlay = *nlay + 1;
    long nact2 = (*ncol) * (*nrow) * (*nlay);
    long ncoor2 = nncol * nnrow * 6;
    long nzcorn2 = nncol * nnrow * nnlay * 4;

    if (mode == 2) {

        size_t mpos = 4 + 16 + 24 + ncoor2 * 8 + nzcorn2 * 4 + nact2 * 4 + 16 + 1;
        fseek(fc, mpos, SEEK_SET);
    }

    if (mode == 1) {
        /*
         *----------------------------------------------------------------------------------
         * Subgrids
         *----------------------------------------------------------------------------------
         */

        /*
         *----------------------------------------------------------------------------------
         * Corner lines
         *----------------------------------------------------------------------------------
         */
        logger_info(LI, FI, FU, "Reading coord...");
        if (fread(coordsv, sizeof(double), ncoord, fc) != ncoord)
            return -201;

        logger_info(LI, FI, FU, "Reading coord... done");
        /*
         *----------------------------------------------------------------------------------
         * Z corner values
         *----------------------------------------------------------------------------------
         */

        logger_info(LI, FI, FU, "Reading zcorn... v2");
        if (fread(zcornsv, sizeof(float), nzcorn, fc) != nzcorn)
            return -202;

        logger_info(LI, FI, FU, "Reading zcorn... done");
        /*
         *----------------------------------------------------------------------------------
         * ACTNUMS
         *----------------------------------------------------------------------------------
         */
        logger_info(LI, FI, FU, "Reading actnums...");
        if (fread(actnumsv, sizeof(int), nact, fc) != nact)
            return -203;
        logger_info(LI, FI, FU, "Reading actnums... done");
    }

    /*
     *----------------------------------------------------------------------------------
     * END DATA + additional METADATA
     *----------------------------------------------------------------------------------
     */
    fgets(xtgeoid, 16, fc);
    if (strncmp(xtgeoid, "<XTGGRIDCP3D_V1", 15) != 0)
        return -300;

    fgets(xtgeoid, 1, fc);  // \n

    logger_info(LI, FI, FU, "Reading metadata... %d", sizeof(metadata));
    char meta[METALEN];
    fgets(meta, METALEN, fc);
    strncpy(metadata, meta, METALEN);
    logger_info(LI, FI, FU, "Reading metadata... done");

    return EXIT_SUCCESS;
}