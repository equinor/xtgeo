/*
 ***************************************************************************************
 *
 * NAME:
 *    grd3d_adj_cells.c
 *
 * DESCRIPTION:
 *    Look through a discrete grid property value, and compare it against
 *    another value. If cells are overlapping somehow, it should be marked.
 *
 * ARGUMENTS:
 *    nx,ny,nz       i     Grid dimensions I J K in input
 *     coordv       i     Grid Z coord for input
 *    zcornsv        i     Grid Z corners for input
 *    actnumsv       i     Grid ACTNUM parameter input
 *    p_prop1        i     Grid Z coord for output
 *    nprop          i     Number of cells in gridprop (shall be nx*ny*nz)
 *    val1           i     value1 (basic value)
 *    val2           i     value to compare vs
 *    p_prop2        o     Resulting property; shall be one when criteria ok
 *    iflag1         i     Options flag (0, use only active cells, 1 all cells)
 *    iflag2         i     0 do not check faults, 1 discard faulted cells
 *
 * RETURNS:
 *    The C macro EXIT_SUCCESS unless problems + changed pointers
 *
 * TODO/ISSUES/BUGS:
 *    None known
 *
 * LICENCE:
 *    cf XTGeo License
 ***************************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"
#include "logger.h"

int
grd3d_adj_cells(int ncol,
                int nrow,
                int nlay,
                double *coordsv,
                long ncoordin,
                double *zcornsv,
                long nzcornin,
                int *actnumsv,
                long nactin,
                int *p_prop1,
                long nprop1,
                int val1,
                int val2,
                int *p_prop2,
                long nprop2,
                int iflag1,
                int iflag2)
{

    long ntotv[3] = { nactin, nprop1, nprop2 };
    if (x_verify_vectorlengths(ncol, nrow, nlay, ncoordin, nzcornin, ntotv, 3) != 0) {
        throw_exception("Errors in array lengths checks in grd3d_adj_cells");
        return EXIT_FAILURE;
    }
    long ib, nnc[6];
    int icn, jcn, kcn, nni, faulted;
    int *useactnum;

    useactnum = calloc(nprop1, sizeof(int));

    for (ib = 0; ib < nprop1; ib++) {
        if (iflag1 == 0)
            useactnum[ib] = actnumsv[ib];
        if (iflag1 == 1)
            useactnum[ib] = 1;
        p_prop2[ib] = 0;
    }

    for (kcn = 1; kcn <= nlay; kcn++) {
        for (jcn = 1; jcn <= nrow; jcn++) {
            for (icn = 1; icn <= ncol; icn++) {

                ib = x_ijk2ib(icn, jcn, kcn, ncol, nrow, nlay, 0);
                if (ib < 0) {
                    free(useactnum);
                    throw_exception("Loop through grid resulted in index outside "
                                    "boundary in grd3d_adj_cells");
                    return EXIT_FAILURE;
                }
                if (useactnum[ib] != 1)
                    continue;
                if (p_prop1[ib] != val1)
                    continue;

                for (nni = 0; nni < 6; nni++)
                    nnc[nni] = -1;
                if (icn > 1)
                    nnc[0] = x_ijk2ib(icn - 1, jcn, kcn, ncol, nrow, nlay, 0);
                if (icn < ncol)
                    nnc[1] = x_ijk2ib(icn + 1, jcn, kcn, ncol, nrow, nlay, 0);
                if (jcn > 1)
                    nnc[2] = x_ijk2ib(icn, jcn - 1, kcn, ncol, nrow, nlay, 0);
                if (jcn < nrow)
                    nnc[3] = x_ijk2ib(icn, jcn + 1, kcn, ncol, nrow, nlay, 0);
                if (kcn > 1)
                    nnc[4] = x_ijk2ib(icn, jcn, kcn - 1, ncol, nrow, nlay, 0);
                if (kcn < nlay)
                    nnc[5] = x_ijk2ib(icn, jcn, kcn + 1, ncol, nrow, nlay, 0);

                for (nni = 0; nni < 6; nni++) {
                    if (nnc[nni] < 0)
                        continue;
                    if (useactnum[nnc[nni]] && p_prop1[nnc[nni]] == val2) {
                        if (p_prop2[ib] < 1)
                            p_prop2[ib] = 1;
                        /* check if the two cells are faulted in XY */
                        if (iflag2 > 0 && nni < 4) {
                            faulted = grd3d_check_cell_splits(ncol, nrow, nlay, coordsv,
                                                              zcornsv, ib, nnc[nni]);
                            if (faulted == 1)
                                p_prop2[ib] = 2;
                        }
                    }
                }
            }
        }
    }

    free(useactnum);

    return EXIT_SUCCESS;
}
