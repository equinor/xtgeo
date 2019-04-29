/*
 *******************************************************************************
 *
 * Modify the ACTNUM based oncell thickness less than a givne threshold
 *
 *******************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"

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

void grd3d_inact_by_dz(
		       int nx,
		       int ny,
		       int nz,
		       double *p_zcorn_v,
		       int   *p_actnum_v,
		       double threshold,
		       int   flip,
		       int   debug
		       )

{
    /* locals */
    int    i, j, k, ib, ndone;
    char   s[24] = "grd3d_inact_by_dz";
    double  *p_dztmp_v;

    xtgverbose(debug);
    xtg_speak(s, 2, "Entering routine...");
    xtg_speak(s, 3, "NX NY NZ: %d %d %d", nx, ny, nz);


    xtg_speak(s, 2, "Finding grid DZ parameter...");

    xtg_speak(s, 2, "Allocating memory to pointer");
    p_dztmp_v = calloc(nx * ny * nz, sizeof(double));


    grd3d_calc_dz(nx, ny, nz, p_zcorn_v, p_actnum_v,
		  p_dztmp_v, flip, 0, debug);

    ndone = 0;

    for (k = 1; k <= nz; k++) {
	xtg_speak(s,3,"Finished layer %d of %d",k,nz);
	for (j = 1; j <= ny; j++) {
	    for (i = 1; i <= nx; i++) {

		/* parameter counting */
		ib = x_ijk2ib(i,j,k,nx,ny,nz,0);

		if (p_dztmp_v[ib] < threshold && p_actnum_v[ib] > 0) {
		    p_actnum_v[ib] = 0;
		    ndone++;
		}

	    }
	}
    }

    xtg_speak(s,2,"Number of cells made active was: %d",ndone);

    free(p_dztmp_v);

    xtg_speak(s,2,"Exiting %s",s);
}
