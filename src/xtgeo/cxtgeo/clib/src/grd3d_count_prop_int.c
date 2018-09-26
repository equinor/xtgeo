/*
 * ############################################################################
 * Author: JCR
 * ############################################################################
 * Count number of a given INT property value (within active cells)
 * ############################################################################
 */


#include "libxtg.h"
#include "libxtg_.h"


int grd3d_count_prop_int(
			 int   nxyz,
			 int   *p_nprop_v,
			 int   *p_actnum_v,
			 int   nval,
			 int   debug
			 )
    
{
    /* locals */
    int ib, n;
    char s[24]="grd3d_count_prop_int";

    xtgverbose(debug);
    n=0;
    for (ib = 0; ib < nxyz; ib++) {	
	if (p_nprop_v[ib]==nval && p_actnum_v[ib]==1) {
	    n++;
	}			    
    }
    xtg_speak(s,2,"Exit grd3d_count_prop_int");
    return n;
}

