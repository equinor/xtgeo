/*
 * ############################################################################
 * Linear interpolation between two 3D points
 * x1..z1 first point
 * x2..z2 second point
 * dlen   requested divider or scaler. 0.5 means in the middle
 *x ... vectors of new points
 * JRIV
 * ############################################################################
 * ############################################################################
 */

#include "libxtg.h"
#include "libxtg_.h"

int
x_vector_linint(double x1,
                double y1,
                double z1,
                double x2,
                double y2,
                double z2,
                double dlen,
                double *xn,
                double *yn,
                double *zn)
{

    /*
     * ------------------------------------------------------------------------
     * Some checks
     * ------------------------------------------------------------------------
     */

    if (x1 == x2 && y1 == y2 && z1 == z2) {
        return -9;
    }

    /*
     * ------------------------------------------------------------------------
     * Compute
     * ------------------------------------------------------------------------
     */

    *xn = x1 * (1 - dlen) + x2 * dlen;
    *yn = y1 * (1 - dlen) + y2 * dlen;
    *zn = z1 * (1 - dlen) + z2 * dlen;

    return 0;
}
