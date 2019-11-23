/*
 * ############################################################################
 * pol_set_entry.c
 *
 * Description:
 * Set x y z entry no i
 *
 * Bugs or potential problems: Allocation trouble, checks needed?
 *
 * Author: J.C. Rivenaes JRIV@statoil.com
 * ############################################################################
 */

#include "libxtg.h"

int pol_set_entry (
		   int    i,
		   double x,
		   double y,
		   double z,
		   int    npmax,
		   double *p_x_v,
		   double *p_y_v,
		   double *p_z_v,
		   int    option,
		   int    debug
		   )
{

    xtgverbose(debug);

    char s[24]="pol_set_entry";
    xtg_speak(s,2,"Entering routine %s ...",s);

    if (i>=npmax) {
	return 0;
    }

    p_x_v[i]=x;
    p_y_v[i]=y;
    p_z_v[i]=z;

    return(1);
}
