/*
 ******************************************************************************
 *
 * SINFO: Compute geometrics as azimuth, md based on X Y Z vectors
 *
 ******************************************************************************
 */

#include <math.h>
#include "libxtg.h"
#include "libxtg_.h"

/*
 ******************************************************************************
 *
 * NAME:
 *    well_geometrics.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Given a trajectory with X Y Z coordinsates, compute the avg angle etc
 *    and returns those arrays
 *
 * ARGUMENTS:
 *    xv             i     x vector np points
 *    yv             i     y vector np points
 *    zv             i     z vector np points
 *    np             i     Number of points
 *    md             o     md vector
 *    az             o     Azimuth; azimith is in degrees, with hor.
 *                         path as 90 degrees
 *    option         i     Options: for future usage
 *    debug          i     Debug flag
 *
 * RETURNS:
 *    Function:  0: Upon success. If problems:
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    Statoil property
 ******************************************************************************
 */


#include "libxtg.h"
#include "libxtg_.h"


int well_geometrics (
                     int np,
                     double *xv,
                     double *yv,
                     double *zv,
                     double *md,
                     double *incl,
                     int option,
                     int debug
                     )
{
    /* locals */
    char s[24] = "well_geometrics";
    int i;
    double incl1, incl2, zdiff;

    xtgverbose(debug);
    xtg_speak(s,3,"Entering routine %s", s);

    for (i = 0; i < np; i ++) {
        if (i > 0) {
            md[i] = md[i - 1] + sqrt(pow((xv[i] - xv[i - 1]), 2.0) +
                                     pow((yv[i] - yv[i - 1]), 2.0) +
                                     pow((zv[i] - zv[i - 1]), 2.0));
        }
        else{
            md[i] = 0.0;
        }


        if (i > 0 && i < (np - 1)) {
            zdiff = fabs(zv[i] - zv[i - 1]);
            if (zdiff > FLOATEPS) {
                incl1 = atan2(sqrt(pow(xv[i] - xv[i - 1], 2.0) +
                                   pow(yv[i] - yv[i - 1], 2.0)),
                              zv[i] - zv[i - 1]) * (180.0/PI);
            }
            else{
                incl1 = 90.0;
            }

            zdiff = fabs(zv[i] - zv[i + 1]);
            if (zdiff > FLOATEPS) {
                incl2 = atan2(sqrt(pow(xv[i + 1] - xv[i], 2.0) +
                                   pow(yv[i + 1] - yv[i], 2.0)),
                              zv[i + 1] - zv[i]) * (180.0/PI);

            }
            else{
                incl2 = 90.0;
            }

            incl[i] = 0.5 * (incl1 + incl2);

        }
    }

    incl[0] = incl[1];
    incl[np - 1] = incl[np - 2];

    if (debug > 2) {
        for (i = 0; i < np; i ++) {
            xtg_speak(s, 3, "Inclination and MD for pos %d: %f   %f",
                      i, incl[i], md[i]);
        }
    }

    return 0;
}
