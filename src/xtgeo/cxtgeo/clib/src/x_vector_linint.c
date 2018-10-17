/*
 * ############################################################################
 * Linear interpolation between two 3D points
 * x1..z1 first point
 * x2..z2 second point
 * dlen   requested divider or scaler. 0.5 means in the middle
 * *x ... vectors of new points
 * JRIV
 * ############################################################################
 * ############################################################################
 */



#include "libxtg.h"
#include "libxtg_.h"


int x_vector_linint (
		    double x1,
		    double y1,
		    double z1,
		    double x2,
		    double y2,
		    double z2,
		    double dlen,
		    double *xn,
		    double *yn,
		    double *zn,
		    int   debug
		    )
{
    /* locals */
    char     sub[24]="x_vector_linint";

    xtgverbose(debug);
    if (debug > 2) xtg_speak(sub, 3, "Entering routine");

    /*
     * ------------------------------------------------------------------------
     * Some checks
     * ------------------------------------------------------------------------
     */

    if ( x1 == x2 && y1 == y2 && z1 == z2) {
	xtg_speak(sub, 2, "Hmmm null length vector");
	return -9;
    }


    /*
     * ------------------------------------------------------------------------
     * Compute
     * ------------------------------------------------------------------------
     */


    if (debug > 2) xtg_speak(sub, 3, "DLEN is %3.2f, x1 is %9.2f   x2 "
                             "is %9.2f", dlen, x1, x2);


    *xn = x1 * (1 - dlen) + x2 * dlen;
    *yn = y1 * (1 - dlen) + y2 * dlen;
    *zn = z1 * (1 - dlen) + z2 * dlen;

    if (debug > 2) xtg_speak(sub, 3, "DLEN is %3.2f, x1 is %9.2f   x2 "
                             "is %9.2f result: is %9.2f", dlen, x1, x2, *xn);


    return 0;
}
