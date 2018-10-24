/*
 * ############################################################################
 * Author: JRIV
 * ############################################################################
 * Copy from one float or double array to another
 * ############################################################################
 */


#include "libxtg.h"
#include "libxtg_.h"


void grd3d_copy_prop_float(
			   long    nxyz,
			   float   *p_input_v,
			   float   *p_output_v,
			   int     debug
			   )

{
    /* locals */
    long ib;
    char s[24]="grd3d_copy_prop_float";

    xtgverbose(debug);
    for (ib = 0; ib < nxyz; ib++) {
	p_output_v[ib]=p_input_v[ib];
    }

    xtg_speak(s,2,"Exit copy float array");
}


void grd3d_copy_prop_doflo(
			   long     nxyz,
			   double  *p_input_v,
			   float   *p_output_v,
			   int     debug
			   )

{
    /* locals */
    long ib;
    char s[24]="grd3d_copy_prop_doflo";

    xtgverbose(debug);
    for (ib = 0; ib < nxyz; ib++) {
	p_output_v[ib] = (float)p_input_v[ib];
    }

    xtg_speak(s,2,"Exit copy float array");
}



void grd3d_copy_prop_double(
			   long     nxyz,
			   double   *p_input_v,
			   double   *p_output_v,
			   int      debug
			   )

{
    /* locals */
    long ib;
    char s[24]="grd3d_copy_prop_double";

    xtgverbose(debug);
    for (ib = 0; ib < nxyz; ib++) {
	p_output_v[ib]=p_input_v[ib];
    }

    xtg_speak(s,2,"Exit copy double array");
}
