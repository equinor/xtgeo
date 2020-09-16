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
#include "roffstuff.h"
#include <math.h>

// TODO: subgrids

static int
_lineshift(int *counter, int limit1, long limit2)
{
    // for line shifting when ascii mode
    (*counter)++;

    int adder = 10;
    if (*counter >= limit1 || *counter >= limit2) {
        adder = 0;
        *counter = 0;
    }
    return adder;
}

void
grdcp3d_export_roff_grid(int mode,
                         int ncol,
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

    strwrite(mode, "tag^translate$", fc);

    strwrite(mode, "float^xoffset^", fc);
    fltwrite(mode, xoffset, fc);

    strwrite(mode, "float^yoffset^", fc);
    fltwrite(mode, yoffset, fc);

    strwrite(mode, "float^zoffset^", fc);
    fltwrite(mode, zoffset, fc);

    strwrite(mode, "endtag$", fc);
    strwrite(mode, "tag^scale$", fc);
    float xscale = 1.0;
    float yscale = 1.0;
    float zscale = -1.0;

    strwrite(mode, "float^xscale^", fc);
    fltwrite(mode, xscale, fc);

    strwrite(mode, "float^yscale^", fc);
    fltwrite(mode, yscale, fc);

    strwrite(mode, "float^zscale^", fc);
    fltwrite(mode, zscale, fc);

    strwrite(mode, "endtag$", fc);
    logger_info(LI, FI, FU, "Initial part... done");

    long nncol = ncol + 1;
    long nnrow = nrow + 1;
    long nnlay = nlay + 1;

    /*
     *----------------------------------------------------------------------------------
     * Corner lines
     *----------------------------------------------------------------------------------
     */

    strwrite(mode, "tag^cornerLines$", fc);
    strwrite(mode, "array^float^data^", fc);
    int myint = (ncol + 1) * (nrow + 1) * 2 * 3;
    intwrite(mode, myint, fc);

    logger_info(LI, FI, FU, "Corner lines... %d", myint);

    long icol, jrow;
    long icc = 0;
    for (icol = 0; icol < nncol; icol++) {
        for (jrow = 0; jrow < nnrow; jrow++) {
            float xtop = (coordsv[icc++] / xscale) - xoffset;
            float ytop = (coordsv[icc++] / yscale) - yoffset;
            float ztop = (coordsv[icc++] / zscale) - zoffset;
            float xbot = (coordsv[icc++] / xscale) - xoffset;
            float ybot = (coordsv[icc++] / yscale) - yoffset;
            float zbot = (coordsv[icc++] / zscale) - zoffset;
            fltwrite(mode + 10, xbot, fc);
            fltwrite(mode + 10, ybot, fc);
            fltwrite(mode + 10, zbot, fc);
            fltwrite(mode + 10, xtop, fc);
            fltwrite(mode + 10, ytop, fc);
            fltwrite(mode, ztop, fc);
        }
    }
    strwrite(mode, "endtag$", fc);
    logger_info(LI, FI, FU, "Corner lines... done");

    /*
     *----------------------------------------------------------------------------------
     * Z corner values
     *----------------------------------------------------------------------------------
     */

    logger_info(LI, FI, FU, "ZCorners...");

    strwrite(mode, "tag^zvalues$", fc);
    strwrite(mode, "array^byte^splitEnz^", fc);
    int ntot = (ncol + 1) * (nrow + 1) * (nlay + 1);
    int ntotcell = ncol * nrow * nlay;
    intwrite(mode, ntot, fc);
    /*
     *----------------------------------------------------------------------------------
     * Z corner values
     *----------------------------------------------------------------------------------
     */

    float *znodes;
    znodes = calloc(ntot * 4, sizeof(float));

    icc = 0;
    long izz = 0;

    int cnt = 0;
    for (icol = 0; icol < nncol; icol++) {
        for (jrow = 0; jrow < nnrow; jrow++) {
            long klay;
            for (klay = nlay; klay >= 0; klay--) {
                int node;
                float znode[4];
                int splitenz = 1;
                float znodeavg = 0.0;
                for (node = 0; node < 4; node++) {
                    long ino = 4 * (icol * nnrow * nnlay + jrow * nnlay + klay) + node;
                    znode[node] = zcornsv[ino];
                    znodeavg += 0.25 * znode[node];
                }
                for (node = 0; node < 4; node++) {
                    if (fabs(znode[node] - znodeavg) > FLOATEPS)
                        splitenz = 4;
                }

                // splitnode always 4 at edges
                if (icol == 0 || jrow == 0 || icol == ncol || jrow == nrow)
                    splitenz = 4;

                int add = _lineshift(&cnt, 12, ntot);
                boolwrite(mode + add, splitenz, fc);

                if (splitenz == 4) {
                    int inode;
                    for (inode = 0; inode < 4; inode++)
                        znodes[izz++] = znode[inode] / zscale - zoffset;
                } else {
                    znodes[izz++] = znode[0] / zscale - zoffset;
                }
            }
        }
    }

    long nznodes = izz;
    logger_info(LI, FI, FU, "nznodes (izz): %ld %d", nznodes, izz);

    strwrite(mode, "array^float^data^", fc);
    myint = (int)nznodes;
    intwrite(mode, myint, fc);
    cnt = 0;
    for (izz = 0; izz < nznodes; izz++) {
        float zn = znodes[izz];
        int add = _lineshift(&cnt, 4, (nznodes - 1));
        fltwrite(mode + add, zn, fc);
    }

    strwrite(mode, "endtag$", fc);
    free(znodes);

    // logger_info(LI, FI, FU, "ZCorners... done");

    /*
     *----------------------------------------------------------------------------------
     * ACTNUMS
     *----------------------------------------------------------------------------------
     */
    logger_info(LI, FI, FU, "ACTNUM...");
    strwrite(mode, "tag^active$", fc);
    strwrite(mode, "array^bool^data^", fc);
    intwrite(mode, (int)ntotcell, fc);

    cnt = 0;
    for (icol = 0; icol < ncol; icol++) {
        for (jrow = 0; jrow < nrow; jrow++) {
            long klay;
            for (klay = (nlay - 1); klay >= 0; klay--) {
                long icc = icol * nrow * nlay + jrow * nlay + klay;

                int add = _lineshift(&cnt, 12, ntotcell);
                boolwrite(mode + add, actnumsv[icc], fc);
            }
        }
    }
    strwrite(mode, "endtag$", fc);
    logger_info(LI, FI, FU, "ACTNUM... done");
}
