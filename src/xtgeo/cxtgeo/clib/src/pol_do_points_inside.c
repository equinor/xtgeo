/*
 ******************************************************************************
 *
 * Do operations inside or outside a polygon for an array of points
 *
 ******************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"
#include <math.h>
/*
 ******************************************************************************
 *
 * NAME:
 *    pol_do_points_inside.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Do operations on Z array for an array of points
 *
 * ARGUMENTS:
 *    xpoi,          i     X coord array
 *    nxpoi          i     Dimension of points (for SWIG)
 *    ypoi,          i     X coord array
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
 *    debug          i     Debug level
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

int pol_do_points_inside(
			 double *xpoi,
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
                         int inside,
                         int debug
			 )

{
    int ic, dowork = 0, istat;
    char sbn[24] = "pol_do_points_inside";

    /*
     *-------------------------------------------------------------------------
     * Loop over all points
     *-------------------------------------------------------------------------
     */
    xtg_speak(sbn, 2, "Check points inside a polygon...");


    for (ic = 0; ic < nzpoi; ic++) {
        dowork = 0;
        istat = pol_chk_point_inside(xpoi[ic], ypoi[ic], xpol, ypol, nxpol,
                                     debug);

        if (istat == -9) {
            xtg_warn(sbn, 1, "Polygon is not closed");
            return 1;
        }

        if (istat > 0 && inside == 1) dowork=1;
        if (istat == 0 && inside == 0) dowork=1;

        if (dowork == 1) {
            if (option == 1) {
                zpoi[ic] = value;
            }
            else if (option == 2) {
                zpoi[ic] += value;
            }
            else if (option == 3) {
                zpoi[ic] -= value;
            }
            else if (option == 4) {
                zpoi[ic] *= value;
            }
            else if (option == 5) {
                if (fabs(value) < FLOATEPS) {
                    zpoi[ic] = UNDEF;
                }
                else{
                    zpoi[ic] /= value;
                }
            }
            else if (option == 11) {
                zpoi[ic] = UNDEF;
            }
            else{
                return 2;
            }
        }
    }
    xtg_speak(sbn, 2, "Check points inside a polygon... done");
    return EXIT_SUCCESS;
}
