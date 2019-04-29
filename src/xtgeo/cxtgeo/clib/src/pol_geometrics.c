/*
 ******************************************************************************
 *
 * SINFO: Compute additonal geometrical vectors from a XYZ polygon
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
 *    pol_geometrics.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Compute various geometrical measures such a length, etc
 *    This can alos be used for well paths!
 *
 * ARGUMENTS:
 *    nlen           i     lenght of vector
 *    xv             i     X array
 *    yv             i     Y array
 *    zv             i     Z array (not used)
 *    hlenv          o     Array describing horizontal cumulative length,
 *                         starting as 0.0 in first point
 *
 * RETURNS:
 *    Function: 0:  upon success. If problems:
 *              -9: Mystical problem
 *
 * TODO/ISSUES/BUGS:
 *    More metrics will be added upon need
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ******************************************************************************
 */

#include <string.h>
#include <stdlib.h>
#include "libxtg.h"
#include "libxtg_.h"

int pol_geometrics(int nlen, double *xv, double *yv, double *zv,
                   double *hlenv, int debug)
{
    int i;
    double hincr;

    char  s[24]="pol_geometrics";

    xtgverbose(debug);

    xtg_speak(s, 2, "Running %s", s);

    for (i=0; i<nlen; i++) {
        if (i > 0) {
            hincr = sqrt(pow(xv[i] - xv[i-1], 2) + pow(yv[i] - yv[i-1], 2));
            hlenv[i] = hlenv[i-1] + hincr;
        }
        else{
            hlenv[i] = 0.0;
        }
    }

    return(0);
}
