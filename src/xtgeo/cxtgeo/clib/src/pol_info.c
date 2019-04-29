/*
 ******************************************************************************
 *
 * SINFO: Get some basic info fro a polyline/gon
 *
 ******************************************************************************
 */

#include <stdlib.h>
#include <math.h>
#include "libxtg.h"

/*
 ******************************************************************************
 *
 * NAME:
 *    pol_info.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Get static geometrical measures such a xmin, xmax, etc
 *
 * ARGUMENTS:
 *    nlen           i     lenght of vector
 *    xv             i     X array
 *    yv             i     Y array
 *    zv             i     Z array
 *    xmin           o     Minimum polygon coordinate X
 *    xmax           o     Maximum polygon coordinate X
 *    ymin           o     Minimum polygon coordinate Y
 *    ymax           o     Maximum polygon coordinate Y
 *    closed         o     1 of polygon i closed, 0 otherwise
 *
 * RETURNS:
 *    0 upon success, -1 if points contains 999.000 or -999.000 in XY
 *    Pointers to output parameters
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ******************************************************************************
 */

int pol_info(int nlen, double *xv, double *yv, double *zv,
             double *xmin, double *xmax, double *ymin, double *ymax,
             int *closed, int debug)
{
    int i;
    double xmi, xma;
    double ymi, yma;

    char s[24] = "pol_info";

    xtgverbose(debug);

    xtg_speak(s, 2, "Running %s", s);

    xmi = VERYLARGEPOSITIVE;
    xma = VERYLARGENEGATIVE;
    ymi = VERYLARGEPOSITIVE;
    yma = VERYLARGENEGATIVE;


    for (i = 0; i < nlen; i++) {

        if (fabs(xv[i]) == 999.00 && fabs(yv[i]) == 999.00) {
            xtg_warn(s, 0, "(%s) 999 entries in polygon; probably a bug", s);
            return -1;
        }

        if (xv[i] < xmi) xmi = xv[i];
        if (xv[i] > xma) xma = xv[i];
        if (yv[i] < ymi) ymi = yv[i];
        if (yv[i] > yma) yma = yv[i];
    }

    *closed = 0;
    if (xv[0] == xv[nlen - 1] && yv[0] == yv[nlen - 1]) *closed = 1;

    *xmin = xmi;
    *xmax = xma;
    *ymin = ymi;
    *ymax = yma;

    return EXIT_SUCCESS;
}
