/*
 ***************************************************************************************
 *
 * NAME:
 *    grd3d_check_cell_splits.c
 *
 * DESCRIPTION:
 *    Check two adjacent cells and report if faulted or not.
 *
 * ARGUMENTS:
 *    nx,ny,nz       i     Grid dimensions I J K in input
 *     coordsv       i     Grid Z coord for input
 *    zcornsv        i     Grid Z corners for input
 *    actnumsv       i     Grid ACTNUM parameter input
 *    ib1            i     Cell count position1
 *    ib2            i     Cell count position2
 *    debug          i     Debug level
 *
 * RETURNS:
 *    Faulted status:
 *    -1: Not resolved, perhaps cells where not aside by side?
 *     0: Cells are unfaulted
 *     1: Cells are faulted
 *
 * TODO/ISSUES/BUGS:
 *    None known
 *
 * LICENCE:
 *    Equinor property
 ***************************************************************************************
 */
#include <math.h>

#include <xtgeo/xtgeo.h>

#include "logger.h"

int
grd3d_check_cell_splits(int ncol,
                        int nrow,
                        int nlay,
                        double *coordsv,
                        double *zcornsv,
                        long ib1,
                        long ib2)
{

    int ic1, ic2, jc1, jc2, kc1, kc2;
    int scase, flag;
    double corners1[24], corners2[24];

    x_ib2ijk(ib1, &ic1, &jc1, &kc1, ncol, nrow, nlay, 0);
    x_ib2ijk(ib2, &ic2, &jc2, &kc2, ncol, nrow, nlay, 0);

    grd3d_corners(ic1, jc1, kc1, ncol, nrow, nlay, coordsv, 0, zcornsv, 0, corners1);

    grd3d_corners(ic2, jc2, kc2, ncol, nrow, nlay, coordsv, 0, zcornsv, 0, corners2);

    scase = 0;
    if (ic2 > 1 && ic1 == ic2 - 1 && jc1 == jc2)
        scase = 1; /* i-1 --> i */
    if (ic2 < ncol && ic1 == ic2 + 1 && jc1 == jc2)
        scase = 2; /* i+1 --> i */
    if (jc2 > 1 && jc1 == jc2 - 1 && ic1 == ic2)
        scase = 3; /* j-1 --> j */
    if (jc2 < nrow && jc1 == jc2 + 1 && ic1 == ic2)
        scase = 4; /* j+1 --> j */

    if (scase == 0)
        return -1; /* could not make/find something useful */

    flag = 0;
    if (scase == 1) {
        if (fabs(corners1[5] - corners2[2]) > FLOATEPS ||
            fabs(corners1[11] - corners2[8]) > FLOATEPS ||
            fabs(corners1[17] - corners2[14]) > FLOATEPS ||
            fabs(corners1[23] - corners2[20]) > FLOATEPS) {

            flag = 1;
        }

    } else if (scase == 2) {
        if (fabs(corners2[5] - corners1[2]) > FLOATEPS ||
            fabs(corners2[11] - corners1[8]) > FLOATEPS ||
            fabs(corners2[17] - corners1[14]) > FLOATEPS ||
            fabs(corners2[23] - corners1[20]) > FLOATEPS) {

            flag = 1;
        }
    } else if (scase == 3) {
        if (fabs(corners1[8] - corners2[2]) > FLOATEPS ||
            fabs(corners1[11] - corners2[5]) > FLOATEPS ||
            fabs(corners1[20] - corners2[14]) > FLOATEPS ||
            fabs(corners1[23] - corners2[17]) > FLOATEPS) {

            flag = 1;
        }
    } else if (scase == 4) {
        if (fabs(corners2[8] - corners1[2]) > FLOATEPS ||
            fabs(corners2[11] - corners1[5]) > FLOATEPS ||
            fabs(corners2[20] - corners1[14]) > FLOATEPS ||
            fabs(corners2[23] - corners1[17]) > FLOATEPS) {

            flag = 1;
        }
    }

    return (flag);
}
