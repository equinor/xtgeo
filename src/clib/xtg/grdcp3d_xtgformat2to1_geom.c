/*
 ***************************************************************************************
 *
 * NAME:
 *    grdcp3d_xtgformat2to1_zcorn.c
 *
 * DESCRIPTION:
 *    Convert ZCORNSV from xtgformat=2 to xtgformat=1 as a compatibility function
 *
 * ARGUMENTS:
 *    ncol, nrow, nlay       i     NCOL, NROW, NLAY dimens
 *    zcornsv2               i     zcorn, on xtgformat == 2 with dimension
 *    zcornsv1               i     zcorn, on xtgformat == 1 with dimension
 *
 * RETURNS:
 *    Function: 0: upon success. If problems:
 *              1: some input points are overlapping
 *              2: the input points forms a line
 *    Result nvector is updated
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ***************************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"
#include "logger.h"

int
grd3cp3d_xtgformat2to1_geom(int ncol,
                            int nrow,
                            int nlay,
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
    int nncol = ncol + 1;
    int nnrow = nrow + 1;
    int nnlay = nlay + 1;

    logger_info(LI, FI, FU, "Dimensions: %d %d %d", ncol, nrow, nlay);
    logger_info(LI, FI, FU, "Transforming grid coordsv -> XTG internal format 2 to 1");
    int j;
    for (j = 0; j < nnrow; j++) {
        int i;
        for (i = 0; i < nncol; i++) {
            int p;
            for (p = 0; p < 6; p++) {
                coordsv1[ib++] = coordsv2[i * nnrow * 6 + j * 6 + p];
            }
        }
    }
    logger_info(LI, FI, FU, "Transforming grid coordsv... done");

    logger_info(LI, FI, FU, "Transforming grid zcornsv -> XTG internal format 2 to 1");

    int k;
    ib = 0;
    for (k = 0; k < nnlay; k++) {  // nnlay is intentional
        int j;
        for (j = 0; j < nrow; j++) {
            int i;
            for (i = 0; i < ncol; i++) {

                long nc0 = 4 * (i * nnrow * nnlay + j * nnlay + k) + 3;
                long nc1 = 4 * ((i + 1) * nnrow * nnlay + j * nnlay + k) + 2;
                long nc2 = 4 * (i * nnrow * nnlay + (j + 1) * nnlay + k) + 1;
                long nc3 = 4 * ((i + 1) * nnrow * nnlay + (j + 1) * nnlay + k + 0);

                zcornsv1[ib++] = zcornsv2[nc0];
                zcornsv1[ib++] = zcornsv2[nc1];
                zcornsv1[ib++] = zcornsv2[nc2];
                zcornsv1[ib++] = zcornsv2[nc3];
            }
        }
    }

    if (ib != nzcorn1) {
        logger_critical(LI, FI, FU, "Something went wrong in %s  IB %ld vs NZCORN1 %ld",
                        FU, ib, nzcorn1);
        return EXIT_FAILURE;
    }

    logger_info(LI, FI, FU, "Transforming grid ROFF zcorn -> XTG representation done");

    logger_info(LI, FI, FU, "Transforming grid actnumsv -> XTG internal format 2 to 1");
    ib = 0;
    for (k = 0; k < nlay; k++) {
        int j;
        for (j = 0; j < nrow; j++) {
            int i;
            for (i = 0; i < ncol; i++) {
                actnumsv1[ib++] = actnumsv2[i * nrow * nlay + j * nlay + k];
            }
        }
    }
    logger_info(LI, FI, FU, "Transforming grid actnumsv... done");

    return EXIT_SUCCESS;
}
