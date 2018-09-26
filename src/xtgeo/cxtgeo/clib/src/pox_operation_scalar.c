/*
 * ############################################################################
 * pox_operation_scalar.c
 *
 * Description:
 * Do scalar operations (add, subtract, etc) between points and scalar,
 * leaving the result in points.
 *
 * iop=1  add
 * iop=2  subtract
 * iop=3  multiply
 * iop=4  divide
 *
 * Bugs or potential problems:
 *
 * The routine return updated versions of p_*1_v and the number of points
 *
 * Author: J.C. Rivenaes
 * ############################################################################
 */

#include <math.h>
#include "libxtg.h"
#include "libxtg_.h"

int pox_operation_scalar (
			  double *p_x1_v,
			  double *p_y1_v,
			  double *p_z1_v,
			  int     np1,
			  double  value,
			  int     iop,
			  int     debug
			  )
{

    int     i;

    char s[24]="pox_operation_scalar";
    xtg_speak(s,2,"Entering routine...");

    xtgverbose(debug);

    if (iop==1) {
	xtg_speak(s,2,"Adding ...");
    }
    else if (iop==2) {
	xtg_speak(s,2,"Subtracting ...");
    }
    else if (iop==3) {
	xtg_speak(s,2,"Multiply ...");
    }
    else if (iop==4) {
	xtg_speak(s,2,"Divide ...");
    }
    else {
	xtg_error(s,"Illegal operation! STOP!");
    }

    for (i=0;i<np1;i++) {
	if (p_x1_v[i] < UNDEF_LIMIT) {
	    /* add */
	    if (iop==1) {
		p_z1_v[i]=p_z1_v[i] + value;
	    }

	    /* subtract */
	    if (iop==2) {
		p_z1_v[i]=p_z1_v[i] - value;
	    }

	    /* multiply */
	    if (iop==3) {
		p_z1_v[i]=p_z1_v[i] * value;
	    }

	    /* divide */
	    if (iop==4) {
		/* avoid division on zero */
		if (fabs(value) < FLOATEPS) value=FLOATEPS;
		p_z1_v[i]=p_z1_v[i] / value;
	    }

	}
    }

    /* return number of points */

    xtg_speak(s,2,"Exiting ...");
    return(np1);

}
