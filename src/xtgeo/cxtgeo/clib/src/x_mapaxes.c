/*
 * ############################################################################
 * x_mapaxes.c
 * ############################################################################
 */

#include <math.h>
#include "libxtg.h"
#include "libxtg_.h"

/*
 * ****************************************************************************
 *                              MAPAXES
 * ****************************************************************************
 * Based on a formula from ERT
 * MAPAXES has 6 elements, and encourage the growth of donkey ears...:
 * ----------------------------------------------------------------------------
 *
 */

void x_mapaxes (
		int   mode,
		double *x,
		double *y,
		const double x1,
		const double y1,
		const double x2,
		const double y2,
		const double x3,
		const double y3,
		int   option,
		int   debug
		)
{
    double  xval, yval, xnew, ynew;
    double  normx, normy;
    double  xx1, xx3, yy1, yy3;
    char    sbn[24] = "x_mapaxes";

    xtgverbose(debug);
    xtg_speak(sbn, 2, "Enter %s", sbn);

    /* mode < 0 means no mapaxes transform */
    if (mode < 0) {
	return;
    }

    xval = *x;
    yval = *y;

    /*
     * Borrowed from ERT method
     * (ref github.com/Ensembles/ert/blob/master/devel/libecl/src/point.c)
     * + ... ecl_grid.c
     */

    xx1 = x1 - x2;
    yy1 = y1 - y2;
    xx3 = x3 - x2;
    yy3 = y3 - y2;

    normx = 1.0 / sqrt(xx3 * xx3 + yy3 * yy3);
    normy = 1.0 / sqrt(xx1 * xx1 + yy1 * yy1);

    xx3 = xx3 * normx;
    yy3 = yy3 * normx;
    xx1 = xx1 * normy;
    yy1 = yy1 * normy;

    xnew = x2 + xval * xx3 + yval * xx1;
    ynew = y2 + xval * yy3 + yval * yy1;

    *x = xnew;
    *y = ynew;

    return;
}
