/*
* #############################################################################
* Name:      grd3d_calc_dz.c
 * Author:    JRIV@statoil.com
 * Created:   2001-12-12
 * Updates:   2016-10-19 include actnum
 * ############################################################################
 * Calculates DZ per cell
 *
 * Arguments:
 *     nx..nz           grid dimensions
 *     p_zcorn_v        ZCORN array (pointer) of input
 *     p_actnum_v       ACTNUM array (pointer) of input
 *     p_dz_v           array (pointer) with DZ values of output
 *     flip             Flag for flipped option, use 1 or -1
 *     option           0: all cells, 1 make actnum 0 cells to undef value
 *     debug            debug/verbose flag
 *
 * Caveeats/issues:
 *
 * ############################################################################
 */

#include "libxtg.h"
#include "libxtg_.h"


void grd3d_calc_dz(
		   int     nx,
		   int     ny,
		   int     nz,
		   double  *p_zcorn_v,
		   int     *p_actnum_v,
		   double  *p_dz_v,
		   int     flip,
		   int     option,
		   int     debug
		   )

{
    /* locals */
    int     i, j, k, ib, ip, iq;
    double  top_z_avg, bot_z_avg;
    char    s[24]="grd3d_calc_dz";

    xtgverbose(debug);
    xtg_speak(s,2,"Entering <grd3d_calc_dz>");
    xtg_speak(s,3,"NX NY NZ: %d %d %d", nx, ny, nz);


    xtg_speak(s,2,"Finding grid DZ parameter...");

    for (k = 1; k <= nz; k++) {
	xtg_speak(s,3,"Finished layer %d of %d",k,nz);
	for (j = 1; j <= ny; j++) {
	    for (i = 1; i <= nx; i++) {

		/* parameter counting */
		ib=x_ijk2ib(i,j,k,nx,ny,nz,0);

		/* grid */
		ip=x_ijk2ib(i,j,k,  nx,ny,nz+1,0);
		iq=x_ijk2ib(i,j,k+1,nx,ny,nz+1,0);

		/* each cell */
		top_z_avg=0.25*(p_zcorn_v[4*ip + 1 - 1]+
				p_zcorn_v[4*ip + 2 - 1]+
				p_zcorn_v[4*ip + 3 - 1]+
				p_zcorn_v[4*ip + 4 - 1]);
		bot_z_avg=0.25*(p_zcorn_v[4*iq + 1 - 1]+
				p_zcorn_v[4*iq + 2 - 1]+
				p_zcorn_v[4*iq + 3 - 1]+
				p_zcorn_v[4*iq + 4 - 1]);

		p_dz_v[ib]=(double)flip*(bot_z_avg - top_z_avg);
		// will do it correct for flipped grids

		if (option==1 && p_actnum_v[ib]==0) {
		    p_dz_v[ib]=UNDEF;
		}

                if (debug > 2 && p_actnum_v[ib]==1) {
                    xtg_speak(s, 3, "Value is %f actnum is %d",
                              p_dz_v[ib], p_actnum_v[ib]);
                }

		if (p_dz_v[ib]<0.0) {
		    xtg_warn(s,1,"Negative dZ for cell %d %d %d ...\n",i,j,k);
		    xtg_warn(s,1,"(Flip status is %d)\n",flip);
		    xtg_warn(s,3,"TOP   1      2     3     4\n");
		    xtg_warn(s,3,"      %8.2f %8.2f %8.2f %8.2f\n",
			     p_zcorn_v[4*ip + 1 - 1], p_zcorn_v[4*ip + 2 - 1],
			     p_zcorn_v[4*ip + 3 - 1], p_zcorn_v[4*ip + 4 - 1]);
		    xtg_warn(s,3,"BOT   1      2     3     4\n");
		    xtg_warn(s,3,"      %8.2f %8.2f %8.2f %8.2f\n",
			     p_zcorn_v[4*iq + 1 - 1], p_zcorn_v[4*iq + 2 - 1],
			     p_zcorn_v[4*iq + 3 - 1], p_zcorn_v[4*iq + 4 - 1]);

		}

	    }
	}
    }
    xtg_speak(s,2,"Exiting <grd3d_calc_dz>");
}
