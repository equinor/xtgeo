
/*
 ******************************************************************************
 *
 * NAME:
 *    pol_do_points_inside.c
 *
 *
 * DESCRIPTION:
 *    Do operations on Z array for an array of points
 *
 * ARGUMENTS:
 *    xpoi,          i     X coord array
 *    nxpoi          i     Dimension of points (for SWIG)
 *    ypoi,          i     Y coord array
 *    nypoi          i     Dimension of points (for SWIG)
 *    zpoi,         i/o    Z coord array
 *    nzpoi          i     Dimension of points (for SWIG)
 *    xpol,          i     X coord array for polygon
 *    nxpol          i     Dimension of polygon (for SWIG)
 *    xpol,          i     X coord array for polygon
 *    nxpol          i     Dimension of polygon (for SWIG)
 *    value          i     value to set/add/ etc
 *    option         i     Options flag of what to do with new value
 *                         1: set; 2: add; 3; subtract; 4: mul, 5: div
 *                         11: eli
 *    inside         i     inside flag: 1= True for inside; 0: outside
 *
 * RETURNS:
 *    Function: 0: upon success. If problems:
 *              1: Polygon is not closed
 *              2: option is not supported
 *    Result nvector is updated
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ******************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"
#include "logger.h"
#include <math.h>

int
pol_do_points_inside(double *xpoi,
                     long nxpoi,
                     double *ypoi,
                     long nypoi,
                     double *zpoi,
                     long nzpoi,
                     double *xpol,
                     long nxpol,
                     double *ypol,
                     long nypol,
                     double value,
                     int option,
                     int inside)

{
    int ic, dowork = 0, istat;

    /*
     *-------------------------------------------------------------------------
     * Loop over all points
     *-------------------------------------------------------------------------
     */

    for (ic = 0; ic < nzpoi; ic++) {
        dowork = 0;
        istat = pol_chk_point_inside(xpoi[ic], ypoi[ic], xpol, ypol, nxpol);

        if (istat == -9) {
            logger_warn(LI, FI, FU, "Polygon is not closed");
            return 1;
        }

        if (istat > 0 && inside == 1)
            dowork = 1;
        if (istat == 0 && inside == 0)
            dowork = 1;

        if (dowork == 1) {
            if (option == 1) {
                zpoi[ic] = value;
            } else if (option == 2) {
                zpoi[ic] += value;
            } else if (option == 3) {
                zpoi[ic] -= value;
            } else if (option == 4) {
                zpoi[ic] *= value;
            } else if (option == 5) {
                if (fabs(value) < FLOATEPS) {
                    zpoi[ic] = UNDEF;
                } else {
                    zpoi[ic] /= value;
                }
            } else if (option == 11) {
                zpoi[ic] = UNDEF;
            } else {
                return 2;
            }
        }
    }
    return EXIT_SUCCESS;
}
