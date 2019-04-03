/*
 * #############################################################################
 * Name:      grd3d_calc_dxdy.c
 * Author:    jriv@statoil.com
 * #############################################################################
 * Compute dx,dy per cell
 *
 * Arguments:
 *     nx..nz           grid dimensions
 *     p_coord_v        Coordinates
 *     p_zcorn_v        ZCORN array (pointer) of input
 *     p_actnum_v       ZCORN array (pointer) of input
 *     dx               DX per cell (output)
 *     dy               DY per cell (output)
 *     option1          0: all cells; 1 set UNDEF if ACTNUM=0
 *     option2          0: not in use
 *     debug            debug/verbose flag
 *
 * Return:
 *     The routine returns an int number stating the success (=0)
 *
 * Caveeats/issues:
 *     DX DY does not match RMS's calculations fully, but I don't know why
 *
 * #############################################################################
 */

#include "libxtg.h"
#include "libxtg_.h"


int grd3d_calc_dxdy(
                    int      nx,
                    int      ny,
                    int      nz,
                    double   *p_coord_v,
                    double   *p_zcorn_v,
                    int      *p_actnum_v,
                    double   *dx,
                    double   *dy,
                    int      option1,
                    int      option2,
                    int      debug
                    )

{
    /* locals */
    int     i, j, k, n, ii, ib;
    char    s[24]="grd3d_calc_dxdy";
    double  c[24], plen, vlen, arad, adeg;

    xtgverbose(debug);

    for (k = 1; k <= nz; k++) {
	xtg_speak(s,3,"Finished layer %d of %d",k,nz);
	for (j = 1; j <= ny; j++) {
	    for (i = 1; i <= nx; i++) {

		ib=x_ijk2ib(i,j,k,nx,ny,nz,0);

                if (option1 == 1 && p_actnum_v[ib] == 0) {
                    dx[ib] = UNDEF;
                    dx[ib] = UNDEF;
                    continue;
                }

                grd3d_corners(i, j, k, nx, ny, nz,
                              p_coord_v, p_zcorn_v, c, debug);

                /* get the length of all lines forming DX */
                plen = 0.0;
                for (n = 0; n <= 3; n++) {
                    ii = 0 + n*6;
                    x_vector_info2(c[ii], c[ii+3], c[ii+1], c[ii+4],
                                   &vlen, &arad, &adeg, 1, debug);
                    plen = plen + vlen;
                }
                dx[ib] = plen/4.0;

                /* get the length of all lines forming DY */
                plen = 0.0;
                for (n = 0; n <= 3; n++) {
                    ii = 0 + n*3;
                    if (n >= 2) ii = 6 + n*3;

                    x_vector_info2(c[ii], c[ii+6], c[ii+1], c[ii+7],
                                   &vlen, &arad, &adeg, 1, debug);
                    plen = plen + vlen;
                }
                dy[ib] = plen/4.0;
            }
        }
    }


    xtg_speak(s,2,"Exit from %s",s);
    return (0);
}
