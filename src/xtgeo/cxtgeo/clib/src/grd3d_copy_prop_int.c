/*
 * ############################################################################
 * Author: JCR
 * ############################################################################
 * Copy from one integer array to another
 * ############################################################################
 */


#include "libxtg.h"
#include "libxtg_.h"


void grd3d_copy_prop_int(
			 long   nxyz,
			 int   *p_input_v,
			 int   *p_output_v,
			 int   debug
			 )

{
    /* locals */
    long ib;
    char s[24]="grd3d_copy_prop_int";

    xtgverbose(debug);
    for (ib = 0; ib < nxyz; ib++) {
	p_output_v[ib]=p_input_v[ib];
    }

    xtg_speak(s,2,"Exit copy integer array");
}
