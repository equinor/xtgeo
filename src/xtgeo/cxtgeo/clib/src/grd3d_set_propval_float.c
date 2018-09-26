/*
 * ############################################################################
 * Author: JCR
 * ############################################################################
 * Set a constant value, all cells
 * ############################################################################
 */


#include "libxtg.h"
#include "libxtg_.h"


void grd3d_set_propval_float(
			     int     nxyz,
			     float   *p_prop_v,
			     float   value,
			     int     debug
			     )
    
{
    /* locals */
    int ib;
    char s[24]="grd3d_set_propval_float";
    
    xtgverbose(debug);

    for (ib = 0; ib < nxyz; ib++) {
	p_prop_v[ib]=value;
    }			    
    
    xtg_speak(s,2,"Exit set prop float value");
}

void grd3d_set_propval_double(
			     int      nxyz,
			     double   *p_prop_v,
			     double   value,
			     int      debug
			     )
    
{
    /* locals */
    int ib;
    char s[24]="grd3d_set_propval_dou.";
    
    xtgverbose(debug);

    for (ib = 0; ib < nxyz; ib++) {
	p_prop_v[ib]=value;
    }			    
    
    xtg_speak(s,2,"Exit set prop double value");
}
