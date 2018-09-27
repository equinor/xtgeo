/*
 * ##################################################################################################
 * Name:      grd3d_calc_z.c
 * Author:    JRIV@statoil.com
 * Created:   2015-09-16
 * Updates:   
 * #################################################################################################
 * Calculates Z (center point depth) per cell
 *
 * Arguments:
 *     nx..nz           grid dimensions
 *     p_zcorn_v        ZCORN array (pointer) of input
 *     p_z_v            array (pointer) with Z values of output
 *     debug            debug/verbose flag
 *
 * Caveeats/issues:
 *     ACTNUM is ignored. A problem or not?
 * #################################################################################################
 */

#include "libxtg.h"
#include "libxtg_.h"


void grd3d_calc_z(
		  int      nx,
		  int      ny,
		  int      nz,
		  double   *p_zcorn_v,
		  double   *p_z_v,
		  int      debug
		  )

{
    /* locals */
    int     i, j, k, ib, ip, iq;
    double  top_z_avg, bot_z_avg;
    char    s[24]="grd3d_calc_z";

    xtgverbose(debug);
    xtg_speak(s,2,"Entering <grd3d_calc_z>");
    xtg_speak(s,3,"NX NY NZ: %d %d %d", nx, ny, nz);


    xtg_speak(s,2,"Finding grid Z parameter...");

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

		p_z_v[ib]=0.5*(bot_z_avg + top_z_avg);
		
	    }	   
	}
    }
    xtg_speak(s,2,"Exiting <grd3d_calc_z>");
}
