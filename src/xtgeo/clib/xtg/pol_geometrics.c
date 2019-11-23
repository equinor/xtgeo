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
 *    This can also be used for well paths!
 *
 * ARGUMENTS:
 *    xv             i     X array
 *    nxv            i     Length of array (for SWIG)
 *    yv             i     Y array
 *    nyv            i     Length of array (for SWIG)
 *    zv             i     Z array (not used)
 *    nzv            i     Length of array (for SWIG)
 *    tlenv          o     Array describing true (3D) cumulative length,
 *                         starting as 0.0 in first point
 *    ntv            i     Length of array (for SWIG)
 *    dtlenv         o     Array describing true delta length from,
 *                         previous point, starting as 0.0 in first point
 *    ndtv            i     Length of array (for SWIG)
 *    hlenv          o     Array describing horizontal cumulative length,
 *                         starting as 0.0 in first point
 *    nhv            i     Length of array (for SWIG)
 *    dhlenv         o     Array describing horizontal delta length from,
 *                         previous point, starting as 0.0 in first point
 *    ndhv            i     Length of array (for SWIG)
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

int pol_geometrics(double *xv,
                   long nxv,
                   double *yv,
                   long nyv,
                   double *zv,
                   long nzv,
                   double *tlenv,
                   long ntv,
                   double *dtlenv,
                   long ndtv,
                   double *hlenv,
                   long nhv,
                   double *dhlenv,
                   long ndhv,
                   int debug)
{
    long i;
    double tincr, hincr;

    char  s[24] = "pol_geometrics";

    xtgverbose(debug);

    xtg_speak(s, 2, "Running %s", s);

    for (i=0; i<nxv; i++) {
        if (i > 0) {
            tincr = sqrt(pow(xv[i] - xv[i-1], 2) + pow(yv[i] - yv[i-1], 2)
                         + pow(zv[i] - zv[i-1], 2));
            dtlenv[i] = tincr;
            tlenv[i] = tlenv[i-1] + tincr;

            hincr = sqrt(pow(xv[i] - xv[i-1], 2) + pow(yv[i] - yv[i-1], 2));
            dhlenv[i] = hincr;
            hlenv[i] = hlenv[i-1] + hincr;
        }
        else{
            dtlenv[i] = 0.0;
            tlenv[i] = 0.0;
            dhlenv[i] = 0.0;
            hlenv[i] = 0.0;
        }
    }

    return(0);
}
