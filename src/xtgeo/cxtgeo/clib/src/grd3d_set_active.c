/*
 * #################################################################################################
 * Name:      grd3d_set_active.c
 * Author:    JRIV@statoil.com
 * Created:   2015-09-17
 * Updates:   
 * #################################################################################################
 * Set all cells within a range active
 *
 * Arguments:
 *     nx..nz           grid dimensions
 *     i1,i2..k2        cell range
 *     p_actnum_v       ACTNUM array (to be modified)
 *     debug            debug/verbose flag
 *
 * Return:              number of cells that has been activated
 *
 * Caveeats/issues:
 *     - if cell thickness is zero, not sure how RMS or Eclipse will deal with that?
 *     - always use zconstent first (with a minimum thickness)?
 * #################################################################################################
 */


#include "libxtg.h"
#include "libxtg_.h"


int grd3d_set_active(
		     int   nx,
		     int   ny,
		     int   nz,
		     int   i1,
		     int   i2,
		     int   j1,
		     int   j2,
		     int   k1,
		     int   k2,
		     int   *p_actnum_v,
		     int   debug
		     )

{
    /* locals */
    int ib, i, j, k, nnn;
    char s[24]="grd3d_set_active";

    xtg_speak(s,2,"Enter %s",s);

    nnn=0;
    xtgverbose(debug);
    for (k = k1; k <= k2; k++) {
	for (j = j1; j <= j2; j++) {
	    for (i = i1; i <= i2; i++) {
		ib=x_ijk2ib(i,j,k,nx,ny,nz,0);
		if (i==9 && j==80 && k==1) {
		    xtg_speak(s,1,"1: Active state %d", p_actnum_v[ib]);
		}

		if (p_actnum_v[ib]==0) {
		    nnn++;
		    p_actnum_v[ib]=1;
		}

		if (i==9 && j==80 && k==1) {
		    xtg_speak(s,1,"2: Active state %d", p_actnum_v[ib]);
		}

	    }
	}
    }			    
 				   
    xtg_speak(s,2,"Exit %s",s);
    return(nnn);
}
