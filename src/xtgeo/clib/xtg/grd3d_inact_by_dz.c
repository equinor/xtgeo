/*
 *******************************************************************************
 *
 * Modify the ACTNUM based oncell thickness less than a givne threshold
 *
 *******************************************************************************
 */

/*
 *******************************************************************************
 *
 * NAME:
 *    grd3d_inact_by_dz.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Set ACTNUM = 0 for cells below a thickness
 *
 * ARGUMENTS:
 *    nx, ny, nz     i     Grid dimensions
 *    p_zcorn_v      i     ZCORN array
 *    p_actnum_v    i/o    ACTNUM array
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
 *******************************************************************************
 */

#include "logger.h"
#include "libxtg.h"
#include "libxtg_.h"


void grd3d_inact_by_dz(
    int nx,
    int ny,
    int nz,
    double *p_zcorn_v,
    long nzcornin,
    int *p_actnum_v,
    long nactin,
    double threshold,
    int   flip
    )

{
    /* locals */
    int    i, j, k, ib, ndone;
    double  *p_dztmp_v;

    p_dztmp_v = calloc(nx * ny * nz, sizeof(double));

    /* lengths of p_zorn etc are dummy */
    grd3d_calc_dz(nx, ny, nz, p_zcorn_v, 0, p_actnum_v, 0,
		  p_dztmp_v, 0, flip, 0);

    ndone = 0;

    for (k = 1; k <= nz; k++) {
	for (j = 1; j <= ny; j++) {
	    for (i = 1; i <= nx; i++) {

		/* parameter counting */
		long ib = x_ijk2ib(i, j, k, nx, ny, nz, 0);
		long ic = x_ijk2ic(i, j, k, nx, ny, nz, 0);

		if (p_dztmp_v[ic] < threshold && p_actnum_v[ic] > 0) {
		    p_actnum_v[ib] = 0;
		    ndone++;
		}

	    }
	}
    }

    free(p_dztmp_v);

}
