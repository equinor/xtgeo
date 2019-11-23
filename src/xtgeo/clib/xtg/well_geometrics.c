/*
****************************************************************************************
 *
 * SINFO: Compute geometrics as azimuth, md based on X Y Z vectors
 *
 ***************************************************************************************
 */

#include <math.h>
#include "libxtg.h"
#include "libxtg_.h"

/*
****************************************************************************************
 *
 * NAME:
 *    well_geometrics.c
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
 *    incl           o     inclination vector in degrees, horizontal is 90 deg
 *    az             o     Azimuth; azimith is in degrees, with hor.
 *                         path as 90 degrees
 *    option         i     Options: for future usage
 *
 * RETURNS:
 *    Function:  0: Upon success. If problems:
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ***************************************************************************************
 */


int well_geometrics (
                     int np,
                     double *xv,
                     double *yv,
                     double *zv,
                     double *md,
                     double *incl,
                     double *az,
                     int option
                     )
{
    /* locals */
    int i;
    double incl1, incl2, zdiff;
    double vlen, arad, adeg1, adeg2;
    double tmp[2];

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

            x_vector_info2(xv[i-1], xv[i], yv[i-1], yv[i], &vlen, &arad,
                           &adeg1, 0, XTGDEBUG);

            zdiff = fabs(zv[i] - zv[i + 1]);
            if (zdiff > FLOATEPS) {
                incl2 = atan2(sqrt(pow(xv[i + 1] - xv[i], 2.0) +
                                   pow(yv[i + 1] - yv[i], 2.0)),
                              zv[i + 1] - zv[i]) * (180.0/PI);

            }
            else{
                incl2 = 90.0;
            }

            x_vector_info2(xv[i], xv[i+1], yv[i], yv[i+1], &vlen, &arad,
                           &adeg2, 0, XTGDEBUG);

            tmp[0] = incl1; tmp[1] = incl2;
            incl[i] = x_avg_angles(tmp, 2);

            tmp[0] = adeg1; tmp[1] = adeg2;
            az[i] = x_avg_angles(tmp, 2);
        }
    }

    incl[0] = incl[1];
    incl[np - 1] = incl[np - 2];

    az[0] = az[1];
    az[np - 1] = az[np - 2];

    return 0;
}
