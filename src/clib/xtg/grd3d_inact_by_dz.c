/*
 ***************************************************************************************
 *
 * NAME:
 *    grd3d_inact_by_dz.c
 *
 *
 * DESCRIPTION:
 *    Set ACTNUM = 0 for cells below a thickness
 *
 * ARGUMENTS:
 *    nx, ny, nz     i     Grid dimensions
 *    zcornsv        i     ZCORN array
 *    actnumsv      i/o    ACTNUM array
 *    threshold      i     Mid cell cell thickness criteria
 *    flip           i     Flip indicator
 *    debug          i     Debug level
 *
 * RETURNS:
 *    Nothing (void)
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

void
grd3d_inact_by_dz(int nx,
                  int ny,
                  int nz,
                  double *zcornsv,
                  long nzcornin,
                  int *actnumsv,
                  long nactin,
                  double threshold,
                  int flip)

{
    /* locals */
    int i, j, k, ndone;
    double *p_dztmp_v;

    p_dztmp_v = calloc(nx * ny * nz, sizeof(double));

    /* lengths of p_zorn etc are dummy */
    grd3d_calc_dz(nx, ny, nz, zcornsv, 0, actnumsv, 0, p_dztmp_v, 0, flip, 0);

    ndone = 0;

    for (k = 1; k <= nz; k++) {
        for (j = 1; j <= ny; j++) {
            for (i = 1; i <= nx; i++) {

                /* parameter counting */
                long ib = x_ijk2ib(i, j, k, nx, ny, nz, 0);
                long ic = x_ijk2ic(i, j, k, nx, ny, nz, 0);
                if (ib < 0 || ic < 0) {
                    free(p_dztmp_v);
                    throw_exception("Loop resulted in index outside "
                                    "boundary in grd3d_inact_by_dz");
                    return;
                }

                if (p_dztmp_v[ic] < threshold && actnumsv[ic] > 0) {
                    actnumsv[ib] = 0;
                    ndone++;
                }
            }
        }
    }

    free(p_dztmp_v);
}
