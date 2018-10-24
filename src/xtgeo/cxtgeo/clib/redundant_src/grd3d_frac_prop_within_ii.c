/*
 * ############################################################################
 * Author: JCR
 * ############################################################################
 * Find fraction of property with another (cell count fraction)
 * ############################################################################
 */


#include "libxtg.h"
#include "libxtg_.h"


double grd3d_frac_prop_within_ii(
			       int   nxyz,
			       int   *p_other_v,
			       int   *p_this_v,
			       int   *p_oarray,
			       int   oalen,
			       int   *p_tarray,
			       int   talen,
			       int   debug
			       )
    
{
    /* locals */
    int ib, nn, mm;
    int ncount_oa, ncount_ow;
    char s[24]="grd3d_frac_prop_.._ii";
    double fraction;
    

    xtgverbose(debug);

    ncount_oa = 0; /* number of all cells of other property (within oarray range)*/
    ncount_ow = 0; /* number of cells of other property (oarray range) sharing with this property (tarray range)*/


    for (ib = 0; ib < nxyz; ib++) {	
	for (nn = 0; nn < oalen; nn++) {
	    if (p_other_v[ib]==p_oarray[nn]) {
		ncount_oa++;
		for (mm = 0; mm < talen; mm++) {
		    if (p_this_v[ib]==p_tarray[mm]) {
			ncount_ow++;
		    }
		}
	    }
	}			    
    }

    fraction=(double)ncount_ow/ncount_oa;

    return(fraction);

    xtg_speak(s,2,"Exit fraction within (int int mode) for properties");
}

