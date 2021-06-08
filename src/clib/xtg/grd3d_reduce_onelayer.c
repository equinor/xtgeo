/*
****************************************************************************************
*
* NAME:
*    grd3d_reduce_onelayer.c
*
* AUTHOR(S):
*
*
* DESCRIPTION:
*    Reduce the grid to one single big layer
*
* ARGUMENTS:
*    nx,ny,nz       i     Grid dimensions I J K in input
*    p_zcorn1_v     i     Grid Z corners for input
*    p_zcorn2_v     o     Grid Z corners for output
*    p_actnum1_v    i     Grid ACTNUM parameter input
*    p_actnum2_v    o     Grid ACTNUM parameter output
*    nactive        o     Number of active cells
*    iflag          i     Options flag (future use)
*    debug          i     Debug level
*
* RETURNS:
*    The C macro EXIT_SUCCESS unless problems + changed pointers
*
* TODO/ISSUES/BUGS:
*    ACTNUM is set to 1 for all cells (iflag=0), only.
*
* LICENCE:
*    cf. XTGeo LICENSE
***************************************************************************************
*/

#include "libxtg.h"
#include "libxtg_.h"
#include "logger.h"

int
grd3d_reduce_onelayer(int nx,
                      int ny,
                      int nz,
                      double *p_zcorn1_v,
                      long nzornin1,
                      double *p_zcorn2_v,
                      long nzornin2,
                      int *p_actnum1_v,
                      long nactin1,
                      int *p_actnum2_v,
                      long nactin2,
                      int *nactive,
                      int iflag)
{
    /* locals */
    int i, j, ic, ib, ibt, ibb, ncc;

    for (j = 1; j <= ny; j++) {
        for (i = 1; i <= nx; i++) {
            /* top */
            ibt = x_ijk2ib(i, j, 1, nx, ny, nz + 1, 0);
            ibb = x_ijk2ib(i, j, 1, nx, ny, 2, 0);
            if (ibt < 0 || ibb < 0) {
                throw_exception("Loop resulted in index outside "
                                "boundary in grd3d_reduce_onelayer");
                return EXIT_FAILURE;
            }
            for (ic = 1; ic <= 4; ic++) {
                p_zcorn2_v[4 * ibb + 1 * ic - 1] = p_zcorn1_v[4 * ibt + 1 * ic - 1];
            }

            /* base */
            ibt = x_ijk2ib(i, j, nz + 1, nx, ny, nz + 1, 0);
            ibb = x_ijk2ib(i, j, 2, nx, ny, 2, 0);
            if (ibt < 0 || ibb < 0) {
                throw_exception("Outside boundary in grd3d_reduce_onelayer");
                return EXIT_FAILURE;
            }
            for (ic = 1; ic <= 4; ic++) {
                p_zcorn2_v[4 * ibb + 1 * ic - 1] = p_zcorn1_v[4 * ibt + 1 * ic - 1];
            }
        }
    }

    /* transfer actnum */
    ncc = 0;

    if (iflag == 0) {
        for (ib = 0; ib < nx * ny * 1; ib++) {
            p_actnum2_v[ib] = 1;
            ncc++;
        }
    } else {
        throw_exception("IFLAG other than 0 not implemented in: grd3d_reduce_onelayer");
        return EXIT_FAILURE;
    }

    *nactive = ncc;

    return EXIT_SUCCESS;
}
