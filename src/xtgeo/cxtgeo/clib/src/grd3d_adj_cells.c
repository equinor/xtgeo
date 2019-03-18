/*
 ******************************************************************************
 *
 * NAME:
 *    grd3d_adj_cells.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Look through a discrete grid property value, and compare it against
 *    another value. If cells are overlapping somehow, it should be marked.
 *
 * ARGUMENTS:
 *    nx,ny,nz       i     Grid dimensions I J K in input
 *    p_coord_v      i     Grid Z coord for input
 *    p_zcorn_v      i     Grid Z corners for input
 *    p_actnum_v     i     Grid ACTNUM parameter input
 *    p_prop1        i     Grid Z coord for output
 *    nprop          i     Number of cells in gridprop (shall be nx*ny*nz)
 *    val1           i     value1 (basic value)
 *    val2           i     value to compare vs
 *    p_prop2        o     Resulting property; shall be one when criteria ok
 *    iflag1         i     Options flag (0, use only active cells, 1 all cells)
 *    iflag2         i     0 do not check faults, 1 discard faulted cells
 *    debug          i     Debug level
 *
 * RETURNS:
 *    The C macro EXIT_SUCCESS unless problems + changed pointers
 *
 * TODO/ISSUES/BUGS:
 *    None known
 *
 * LICENCE:
 *    Equinor property
 ******************************************************************************
 */


#include "libxtg.h"
#include "libxtg_.h"


int grd3d_adj_cells (
                     int ncol,
                     int nrow,
                     int nlay,
                     double *p_coord_v,
                     double *p_zcorn_v,
                     int *p_actnum_v,
                     int *p_prop1,
                     long nprop1,
                     int val1,
                     int val2,
                     int *p_prop2,
                     long nprop2,
                     int iflag1,
                     int iflag2,
                     int debug
                     )
{
    /* locals */
    char sbn[24] = "grd3d_adj_cells";
    long ib;
    int icn, jcn, kcn, nni, faulted;
    int useactnum[nprop1], nnc[6];

    for (ib = 0; ib < nprop1; ib++) {
        if (iflag1 == 0) useactnum[ib] = p_actnum_v[ib];
        if (iflag1 == 1) useactnum[ib] = 1;
        p_prop2[ib] = 0;
    }

    xtg_speak(sbn, 2, "First check all cells connections");
    for (kcn = 1; kcn <= nlay; kcn++) {
        for (jcn = 1; jcn <= nrow; jcn++) {
            for (icn = 1; icn <= ncol; icn++) {

                ib = x_ijk2ib(icn, jcn, kcn, ncol, nrow, nlay, 0);
                if (useactnum[ib] != 1) continue;
                if (p_prop1[ib] != val1) continue;

                for (nni = 0; nni < 6; nni++) nnc[nni] = -1;
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
                    if (nnc[nni] < 0) continue;
                    if (useactnum[nnc[nni]] && p_prop1[nnc[nni]] == val2) {
                        if (p_prop2[ib] < 1) p_prop2[ib] = 1;
                        /* check if the two cells are faulted in XY */
                        if (iflag2 > 0 && nni < 4) {
                            faulted = grd3d_check_cell_splits(ncol, nrow, nlay,
                                                              p_coord_v,
                                                              p_zcorn_v,
                                                              ib, nnc[nni],
                                                              debug);
                            if (faulted == 1) p_prop2[ib] = 2;
                        }
                    }
                }
            }
	}
    }
    return EXIT_SUCCESS;

}
