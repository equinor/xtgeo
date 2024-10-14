/*
 ***************************************************************************************
 *
 * NAME:
 *    grdcp3d_xtgformat1to2_zcorn.c
 *
 * DESCRIPTION:
 *    Convert ZCORNSV from xtgformat=1 to xtgformat=2 as a compatibility function
 *
 * ARGUMENTS:
 *    ncol, nrow, nlay       i     NCOL, NROW, NLAY dimens
 *    *v1                    i     zcorn etc, on xtgformat == 1 with dimensions
 *    *v2                    i     zcorn etc, on xtgformat == 2 with dimensions
 *
 * RETURNS:
 *    Function: 0: upon success.
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ***************************************************************************************
 */
#include <stdlib.h>
#include <xtgeo/xtgeo.h>
#include "logger.h"

int
grd3cp3d_xtgformat1to2_geom(long ncol,
                            long nrow,
                            long nlay,
                            double *coordsv1,
                            long ncoordsv1,
                            double *coordsv2,
                            long ncoordsv2,
                            double *zcornsv1,
                            long nzcorn1,
                            float *zcornsv2,  // notice float type
                            long nzcorn2,
                            int *actnumsv1,
                            long nact1,
                            int *actnumsv2,
                            long nact2)

{

    long ib = 0;
    long nncol = ncol + 1;
    long nnrow = nrow + 1;
    long nnlay = nlay + 1;

    logger_info(LI, FI, FU, "Transforming grid coordsv -> XTG internal format 1 to 2");
    long i;
    for (i = 0; i < nncol; i++) {
        long j;
        for (j = 0; j < nnrow; j++) {
            long p;
            for (p = 0; p < 6; p++) {
                coordsv2[ib++] = coordsv1[j * nncol * 6 + i * 6 + p];
            }
        }
    }
    logger_info(LI, FI, FU, "Transforming grid coordsv... done");

    logger_info(LI, FI, FU, "Transforming grid zcornsv -> XTG internal format 1 to 2");

    ib = 0;
    for (i = 0; i < nncol; i++) {
        long j;
        for (j = 0; j < nnrow; j++) {
            long k;
            for (k = 1; k <= nnlay; k++) {

                long ib1 = x_ijk2ib(i + 0, j + 0, k, ncol, nrow, nnlay, 0);
                long ib2 = x_ijk2ib(i + 1, j + 0, k, ncol, nrow, nnlay, 0);
                long ib3 = x_ijk2ib(i + 0, j + 1, k, ncol, nrow, nnlay, 0);
                long ib4 = x_ijk2ib(i + 1, j + 1, k, ncol, nrow, nnlay, 0);

                long sw = 4 * ib1 + 1 * 4 - 1;  // SW node is 4'th corner i, j
                long se = 4 * ib2 + 1 * 3 - 1;  // SE node is 3'rd corner i+1, j
                long nw = 4 * ib3 + 1 * 2 - 1;  // NW node is 2'nd corner i, j+1
                long ne = 4 * ib4 + 1 * 1 - 1;  // NE node is 1'th corner i+1, j+1

                if (i == 0 && j == 0) {
                    sw = ne;
                    se = ne;
                    nw = ne;
                } else if (i == 0 && j > 0 && j < nrow) {
                    sw = se;
                    nw = ne;
                } else if (i == 0 && j == nrow) {
                    sw = se;
                    nw = se;
                    ne = se;
                } else if (i > 0 && i < ncol && j == nrow) {
                    nw = sw;
                    ne = se;
                } else if (i == ncol && j == nrow) {
                    ne = sw;
                    se = sw;
                    nw = sw;
                } else if (i == ncol && j > 0 && j < nrow) {
                    se = sw;
                    ne = nw;
                } else if (i == ncol && j == 0) {
                    sw = nw;
                    se = nw;
                    ne = nw;
                } else if (i > 0 && i < ncol && j == 0) {
                    sw = nw;
                    se = ne;
                }

                zcornsv2[ib++] = (float)zcornsv1[sw];
                zcornsv2[ib++] = (float)zcornsv1[se];
                zcornsv2[ib++] = (float)zcornsv1[nw];
                zcornsv2[ib++] = (float)zcornsv1[ne];
            }
        }
    }
    if (ib != nzcorn2) {
        throw_exception("Error in: grd3cp3d_xtgformat1to2_geom, ib != nzcorn2");
        return EXIT_FAILURE;
    }

    // probably not needed?
    grdcp3d_process_edges(ncol, nrow, nlay, zcornsv2, 0);

    logger_info(LI, FI, FU, "Transforming grid ROFF zcorn -> XTG representation done");

    logger_info(LI, FI, FU,
                "Transforming grid actnumsv -> XTG longernal format 1 to 2");
    ib = 0;
    for (i = 0; i < ncol; i++) {
        long j;
        for (j = 0; j < nrow; j++) {
            long k;
            for (k = 0; k < nlay; k++) {
                actnumsv2[ib++] = actnumsv1[k * ncol * nrow + j * ncol + i];
            }
        }
    }

    logger_info(LI, FI, FU, "Transforming grid actnumsv... done");

    return EXIT_SUCCESS;
}
