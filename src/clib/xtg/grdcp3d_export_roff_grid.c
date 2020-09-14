/*
****************************************************************************************
*
* NAME:
*    grd3d_export_roff_grid.c
*
* DESCRIPTION:
*    Export to ROFF format.  See also: grd3d_export_roff_end
*
*    -----------------------------------------------------------------------------------
*    roff-asc
*     #ROFF file#
*     #Creator: RMS - Reservoir Modelling System, version 6.0#
*     tag filedata
*     int byteswaptest 1
*     char filetype  "grid"
*     char creationDate  "28/03/2000 16:59:16"
*     endtag
*     tag version
*     int major 2
*     int minor 0
*     endtag
*     tag dimensions
*     int ncol 4
*     int nrow 4
*     int nlay 3
*     endtag
*     tag translate
*     float xoffset   4.62994625E+05
*     float yoffset   5.93379900E+06
*     float zoffset  -3.37518921E+01
*     endtag
*     tag scale
*     float xscale   1.00000000E+00
*     float yscale   1.00000000E+00
*     float zscale  -1.00000000E+00
*     endtag
*     tag subgrids
*     array int nLayers 16
*          1            1            5            8           10           10
*         10           15            8           20           20           20
*          8            6            8            2
*    endtag
*    tag cornerLines
*    array float data 150
*     -7.51105194E+01  -4.10773730E+03  -1.86212000E+03  -7.51105194E+01
*     -4.10773730E+03  -1.72856909E+03  -8.36509094E+02  -2.74306006E+03
*     ....
*    endtag
*    tag zvalues
*    array byte splitEnlay 100
*     1   1   1   1   1   1   4   4   1   1   1   1
*    ....
*    endtag
*    tag active
*    array bool data 48
*     1   1   1   1   1   1   1   1   1   1   1   1
*     1   1   1   1   1   1   1   1   1   1   1   1
*     1   1   1   1   1   1   1   1   1   1   1   1
*     1   1   1   1   1   1   1   1   1   1   1   1
*    endtag
*     ... ETC
*
* ARGUMENTS:
*    mode                i     txt or binary
*    ncol, nrow, nlay          i     NCOL, NROW, NLAY
*    num_subgrds         i     Number of subgrids
*    isubgrd_to_export   i     Which subgrd to export
*    *offset             i     coordinate offsets
*    coordsv             i     COORD array w/ len
*    zcornsv             i     ZCORN array w/ len
*    actnumsv            i     ACTNUM array w/ len
*    p_subgrd_v          i     Subgrid array
*    filename            i     File name
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
#include <math.h>

// TODO: subgrids

void
grdcp3d_export_roff_grid(int ncol,
                         int nrow,
                         int nlay,
                         double xoffset,
                         double yoffset,
                         double zoffset,
                         double *coordsv,
                         long ncoordin,
                         float *zcornsv,
                         long nlaycornin,
                         int *actnumsv,
                         long nactin,
                         FILE *fc)

{
    /*
     *----------------------------------------------------------------------------------
     * Initial part
     *----------------------------------------------------------------------------------
     */

    logger_info(LI, FI, FU, "Initial part...");

    fwrite("tag\0translate\0", 1, 14, fc);

    fwrite("float\0xoffset\0", 1, 14, fc);
    float myfloat = xoffset;
    fwrite(&myfloat, 4, 1, fc);

    fwrite("float\0yoffset\0", 1, 14, fc);
    myfloat = yoffset;
    fwrite(&myfloat, 4, 1, fc);

    fwrite("float\0zoffset\0", 1, 14, fc);
    myfloat = zoffset;
    fwrite(&myfloat, 4, 1, fc);
    fwrite("endtag\0", 1, 7, fc);

    fwrite("tag\0scale\0", 1, 10, fc);
    float xscale = 1.0;
    float yscale = 1.0;
    float zscale = -1.0;

    fwrite("float\0xscale\0", 1, 13, fc);
    myfloat = xscale;
    fwrite(&myfloat, 4, 1, fc);

    fwrite("float\0yscale\0", 1, 13, fc);
    myfloat = yscale;
    fwrite(&myfloat, 4, 1, fc);

    fwrite("float\0zscale\0", 1, 13, fc);
    myfloat = zscale;
    fwrite(&myfloat, 4, 1, fc);

    fwrite("endtag\0", 1, 7, fc);
    logger_info(LI, FI, FU, "Initial part... done");

    /*
     *----------------------------------------------------------------------------------
     * Corner lines
     *----------------------------------------------------------------------------------
     */
    logger_info(LI, FI, FU, "Corner lines...");

    fwrite("tag\0cornerLines\0", 1, 16, fc);
    fwrite("array\0float\0data\0", 1, 17, fc);
    int myint = (ncol + 1) * (nrow + 1) * 2 * 3;
    fwrite(&myint, 4, 1, fc);

    long icol, jrow;
    long icc = 0;
    for (icol = 0; icol <= ncol; icol++) {
        for (jrow = 0; jrow <= nrow; jrow++) {
            float xtop = (coordsv[icc++] / xscale) - xoffset;
            float ytop = (coordsv[icc++] / yscale) - yoffset;
            float ztop = (coordsv[icc++] / zscale) - zoffset;
            float xbot = (coordsv[icc++] / xscale) - xoffset;
            float ybot = (coordsv[icc++] / yscale) - yoffset;
            float zbot = (coordsv[icc++] / zscale) - zoffset;
            fwrite(&xbot, 4, 1, fc);
            fwrite(&ybot, 4, 1, fc);
            fwrite(&zbot, 4, 1, fc);
            fwrite(&xtop, 4, 1, fc);
            fwrite(&ytop, 4, 1, fc);
            fwrite(&ztop, 4, 1, fc);
        }
    }
    fwrite("endtag\0", 1, 7, fc);
    logger_info(LI, FI, FU, "Corner lines... done");

    /*
     *----------------------------------------------------------------------------------
     * Z corner values
     *----------------------------------------------------------------------------------
     */

    logger_info(LI, FI, FU, "ZCorners...");

    fwrite("tag\0zvalues\0", 1, 12, fc);
    fwrite("array\0byte\0splitEnz\0", 1, 20, fc);
    int ntot = (ncol + 1) * (nrow + 1) * (nlay + 1);
    int ntotcell = ncol * nrow * nlay;
    fwrite(&ntot, 4, 1, fc);
    /*
     *----------------------------------------------------------------------------------
     * Z corner values
     *----------------------------------------------------------------------------------
     */

    int *splitv;
    splitv = calloc(ntot, sizeof(int));
    float *znodes;
    znodes = calloc(ntot * 4, sizeof(float));

    icc = 0;
    long izz = 0;
    for (icol = 0; icol <= ncol; icol++) {
        for (jrow = 0; jrow <= nrow; jrow++) {
            long klay;
            for (klay = nlay; klay >= 0; klay--) {
                int node;
                float znode[4];
                int splitenz = 1;
                for (node = 0; node < 4; node++) {
                    icc = 4 * (icol * nrow * nlay + jrow * nlay + klay) + node;
                    znode[node] = zcornsv[icc];
                }
                int inode;
                for (inode = 0; inode < 4; inode++) {
                    for (node = 0; node < 4; node++) {
                        if (znode[inode] != znode[node])
                            splitenz = 4;
                    }
                }

                // splitnode always 4 at edges
                if (icol == 0 || jrow == 0 || icol == ncol || jrow == nrow)
                    splitenz = 4;
                fwrite(&splitenz, 4, 1, fc);

                splitv[icc++] = splitenz;
                if (splitenz == 4) {
                    for (inode = 0; inode < 4; inode++)
                        znodes[izz++] = znode[inode] / zscale - zoffset;
                } else {
                    znodes[izz++] = znode[0] / zscale - zoffset;
                }
            }
        }
    }
    fwrite("endtag\0", 1, 7, fc);
    logger_info(LI, FI, FU, "ZCorners... ...");

    long nznodes = izz;
    fwrite("array\0float\0data\0", 1, 17, fc);
    myint = (int)nznodes;
    fwrite(&myint, 4, 1, fc);
    for (izz = 0; izz < nznodes; izz++)
        fwrite(&znodes[izz], 4, 1, fc);

    free(splitv);
    free(znodes);

    /*
     *----------------------------------------------------------------------------------
     * ACTNUMS
     *----------------------------------------------------------------------------------
     */
    fwrite("tag\0active\0", 1, 11, fc);
    fwrite("array\0bool\0data\0", 1, 16, fc);
    myint = (int)ntotcell;
    fwrite(&myint, 4, 1, fc);
    for (icol = 0; icol <= ncol; icol++) {
        for (jrow = 0; jrow <= nrow; jrow++) {
            long klay;
            for (klay = nlay; klay >= 0; klay--) {
                long icc = icol * nrow * nlay + jrow * nlay + klay;
                char mybyte = actnumsv[icc];
                fwrite(&mybyte, 1, 1, fc);
            }
        }
    }
    fwrite("endtag\0", 1, 7, fc);
}
