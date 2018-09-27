/*
 * ############################################################################
 * Author: JCR
 * ############################################################################
 * Set a constant value, all cells
 * ############################################################################
 */


#include "libxtg.h"
#include "libxtg_.h"


void grd3d_set_propval_int(
			     int     nxyz,
			     int     *p_prop_v,
			     int     value,
			     int     debug
			     )
    
{
    /* locals */
    int ib;
    char s[24]="grd3d_set_propval_int";
    
    xtgverbose(debug);
    
    for (ib = 0; ib < nxyz; ib++) {
	p_prop_v[ib]=value;
    }			    
    
    xtg_speak(s,2,"Exit set prop int value");
}
