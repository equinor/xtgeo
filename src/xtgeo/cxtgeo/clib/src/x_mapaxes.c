/*
 * ############################################################################
 * x_mapaxes.c
 * Compute MAPAXES transform. The inputy *x and *y are in scaled coordinates
 * The x1, y1, ....x3, y3 are from MAPAXES. xmin, xmax etc are min and max data
 * found from read data (in scaled domain). *x, *y are updated to be
 * in UTM coordinates from this routine.
 * ############################################################################
 * ############################################################################
 */

#include <math.h>
#include "libxtg.h"
#include "libxtg_.h"

/*
 * ****************************************************************************
 *                              MAPAXES
 * ****************************************************************************
 * X = a*X' + b*Y' + c
 * Y = d*X' + e*Y' + f
 * Based on a formula from T. Barkve, Roxar
 * MAPAXES has 6 elements:
 * x1 y1 x2 y2 x3 y3 (1 is a point on the Y axis, 2 is origo, 3 a point on X)
 * xmin, ymin, xmax, ymax in local coords must be computed from the full grid
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
		double xmin,
		double xmax,
		double ymin,
		double ymax,
		int   option,
		int   debug
		)
{
    double  dx, dy, xval, yval, xnew, ynew, a, b, c, d, e, f;
    double  normx, normy, norm;
    double  xx1, xx3, yy1, yy3;
    char    s[24]="x_mapaxes";

    xtgverbose(debug);

    /* mode = 0 means no mapaxes transform */
    if (mode==0) {
	return;
    }

    /* force this! */
    option = 2;

    xval = *x;
    yval = *y;

    /* Tor Barkve method (does not work for all cases) */
    if (option==0) {
	dx   = xmax - xmin;
	dy   = ymax - ymin;


	if (fabs(dx) > 1.e-10 && fabs(dy) > 1.e-10 ) {
	    a = (x3 - x2)/dx;
	    b = (x1 - x2)/dx;
	    c = x2;
	    d = (y3 - y2)/dy;
	    e = (y1 - y2)/dy;
	    f = y2;
	}
	else{
	    xtg_error(s,"Error in MAPAXES...");
	}

	if (debug > 3) {
	    xtg_speak(s,4,"MAPAXES x1...y3: %11.2f %11.2f %11.2f %11.2f "
                      "%11.2f %11.2f",x1,y1,x2,y2,x3,y3);
	    xtg_speak(s,4,"MAPAXES a ... f: %11.2f %11.2f %11.2f %11.2f "
                      "%11.2f %11.2f",a,b,c,d,e,f);
	}
	xval  = xval   - xmin;
	yval  = yval   - ymin;
	xnew  = a*xval + b*yval + c;
	ynew  = d*xval + e*yval + f;

    }

    /* ERT method
     * (ref github.com/Ensembles/ert/blob/master/devel/libecl/src/point.c) */
    /* + ... ecl_grid.c */
    /*
     * MAPAXES   x1  y1            x2   y2         x3   y3
     *         Y_unit(x,y)        Origo(x,y)      X_unit(x,y)
     *
     * Finally this works!
     */

    else if (option == 2) {

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
    }


    /* JRIV inversion of ERT code... */
    /* MAPAXES   x1  y1            x2   y2       x3   y3
     *          Y_unit(x,y)      Origo(x,y)   X_unit(x,y)
     * Not finished; does not work blabla, keep for reference...
     */
    else if (option == 3) {

	xx1 = x1 - x2;
	yy1 = y1 - y2;
	xx3 = x3 - x3;
	yy3 = y3 - y2;

	normx = 1.0 / sqrt(xx3 * xx3 + yy3 * yy3);
	normy = 1.0 / sqrt(xx1 * xx1 + yy1 * yy1);

	xx3 = xx3 * normx;
	yy3 = yy3 * normx;
	xx1 = xx1 * normy;
	yy1 = yy1 * normy;

	norm = 1.0 / (xx3*yy1 - yy3*xx1);

	/* based on my notes... 20160114*/

	c = (yy3 / xx3 * yy1) * ( xval / norm + yy1 * x2 - y2 * xx1 ) +
            (1.0 / xx3) * ( yval / norm - x2 * yy3 + y2 * xx3);

	ynew = c / (1.0 - (xx1 * yy3 / (xx3 * yy1)));

	xnew = (1.0 / yy1) * (xval / norm + yy1 * x2 + yval * xx1 - y2 * xx1);


    }


    *x = xnew;
    *y = ynew;

    return;
}
