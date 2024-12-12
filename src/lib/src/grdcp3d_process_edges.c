/*
 ***************************************************************************************
 *
 * NAME:
 *    grdcp3d_process_edges.c
 *
 * DESCRIPTION:
 *    The _xtgformat=2 format has 4 nodes per pillar. At edges, there will be "outside"
 *    nodes that in principle can have any values. However, some routines will check eg
 *    splits, and when that, doing an "edge check" is more code which could be avoided
 *    if outside nodes have logical values.
 *
 *    This routine will secure that the 4 nodes on edges are sane as a postprocessing.
 *    Hence, this is done in order to avoid to much special treatment in other routines.
 *
 *                 |
 *          nw     |   ne       Example:
 *               2 | 3          If node is the 0,0 corner, then node 3 is the actual
 *          -------+-------     corner value and node 0..2 shall be set equal to 3
 *               0 | 1
 *           sw    |   se
 *                 |
 *
 *
 * ARGUMENTS:
 *    ncol,nrow,nlay   i      Grid dimensions nx ny nz
 *    zcornsv         i/o     Grid Z corners for input with dimensions for SWIG numpy
 *
 * RETURNS:
 *    Updates zcornsv, _xtgformat=2
 *
 * TODO/ISSUES/BUGS:
 *    None known
 *
 * LICENCE:
 *    cf. XTGeo License
 ***************************************************************************************
 */
#include <xtgeo/xtgeo.h>
#include "logger.h"

void
grdcp3d_process_edges(long ncol, long nrow, long nlay, float *zcornsv, long nzcornsv)

{
    /* each cell is defined by 4 pillars */

    logger_info(LI, FI, FU, "Process zcornsv edges...");
    long i, j, k;
    long nnrow = nrow + 1;
    long nnlay = nlay + 1;

    for (k = 0; k < nnlay; k++) {

        long node;

        // corner i = 0 j = 0
        node = 4 * (nnlay * nnrow * 0 + 0 * nnlay + k);
        zcornsv[node + 0] = zcornsv[node + 3];
        zcornsv[node + 1] = zcornsv[node + 3];
        zcornsv[node + 2] = zcornsv[node + 3];

        // corner i = 0 j = nrow
        node = 4 * (nnlay * nnrow * 0 + nrow * nnlay + k);
        zcornsv[node + 0] = zcornsv[node + 1];
        zcornsv[node + 2] = zcornsv[node + 1];
        zcornsv[node + 3] = zcornsv[node + 1];

        // corner i = ncol j = 0
        node = 4 * (nnlay * nnrow * ncol + 0 * nnlay + k);
        zcornsv[node + 0] = zcornsv[node + 2];
        zcornsv[node + 1] = zcornsv[node + 2];
        zcornsv[node + 3] = zcornsv[node + 2];

        // corner i = ncol j = nrow
        node = 4 * (nnlay * nnrow * ncol + nrow * nnlay + k);
        zcornsv[node + 1] = zcornsv[node + 0];
        zcornsv[node + 2] = zcornsv[node + 0];
        zcornsv[node + 3] = zcornsv[node + 0];

        // i == 0 boundary
        for (j = 1; j < nrow; j++) {
            node = 4 * (nnlay * nnrow * 0 + j * nnlay + k);
            zcornsv[node + 2] = zcornsv[node + 3];
            zcornsv[node + 0] = zcornsv[node + 1];
        }

        // i == ncol boundary
        for (j = 1; j < nrow; j++) {
            node = 4 * (nnlay * nnrow * ncol + j * nnlay + k);
            zcornsv[node + 3] = zcornsv[node + 2];
            zcornsv[node + 1] = zcornsv[node + 0];
        }

        // j == 0 boundary
        for (i = 1; i < ncol; i++) {
            node = 4 * (nnlay * nnrow * i + 0 * nnlay + k);
            zcornsv[node + 0] = zcornsv[node + 2];
            zcornsv[node + 1] = zcornsv[node + 3];
        }

        // j == nrow boundary
        for (i = 1; i < ncol; i++) {
            node = 4 * (nnlay * nnrow * i + nrow * nnlay + k);
            zcornsv[node + 2] = zcornsv[node + 0];
            zcornsv[node + 3] = zcornsv[node + 1];
        }
    }
    logger_info(LI, FI, FU, "Process zcornsv edges... done");
}
