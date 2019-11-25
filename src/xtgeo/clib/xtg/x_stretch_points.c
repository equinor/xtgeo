/*
 * ############################################################################
 * x_stretch_points.c
 *
 * Take two end members of a series of 1D point values, and stretch or squeeze
 * proportionally betweeen these. Set p1 is the input, while p2 is the output
 * p1_v[0]=start p1_v[np-1]=stop, similar for p2_v
 * For p2, numbers between [0] and [np-1] are overwritten.
 * The "method" flag is currently not used.
 *
 * ############################################################################
 * ToDo:
 * - Non knows issues
 * ############################################################################
 */


#include <math.h>
#include "libxtg.h"
#include "libxtg_.h"


void x_stretch_points (
		       int   np,
		       double *p1_v,
		       double *p2_v,
		       int method,
		       int debug
			)
{
    /* locals */
    int i;
    double p1_diff, p2_diff, ratio, p2stop_orig;
    char sub[24]="x_stretch_points";

    xtgverbose(debug);

    xtg_speak(sub,3,"Entering routine");


    /*
     * ########################################################################
     * Some checks
     * ########################################################################
     */
    if ( p1_v[0] > p1_v[np-1] || p2_v[0] > p2_v[np-1]) {
	xtg_error(sub,"Invalid input");
    }

    /* check that p1 is sorted */
    for (i=1;i<np;i++) {
	if (p1_v[i]<p1_v[i-1]) {
	    xtg_error(sub,"Invalid input, array not sorted increasingly");
	}
    }


    /*
     * ########################################################################
     * Work
     * ########################################################################
     */
    p1_diff=p1_v[np-1]-p1_v[0];
    p2_diff=p2_v[np-1]-p2_v[0];

    p2stop_orig=p2_v[np-1];

    /* avoid dividing on zero */
    if (p1_diff<FLOATEPS) p1_diff=FLOATEPS;

    ratio=p2_diff/p1_diff;

    for (i=1;i<np;i++) {
	p2_v[i]=p2_v[i-1]+(p1_v[i]-p1_v[i-1])*ratio;
    }

    if (fabs(p2stop_orig - p2_v[np-1]) > FLOATEPS) {
	xtg_error(sub,"Invalid stretch or squeeze - contact the author");
    }

}
