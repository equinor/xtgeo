/*
 ******************************************************************************
 *
 * SINFO: Flag values where one well path is approx equal to other as undef
 *
 ******************************************************************************
 */

#include "logger.h"
#include <math.h>
#include "libxtg.h"
#include "libxtg_.h"

/*
 ******************************************************************************
 *
 * NAME:
 *    well_trunc_parallel.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Look at one well traj and compare with other. If points are rather equal,
 *    flag them with UNDEF value
 *
 *    In arguments, nx1 etc are repeated for swig typemap to numpy needs
 *
 * ARGUMENTS:
 *    xv1             i     x vector np points
 *    nx1             i     Number of points
 *    yv1             i     y vector np points
 *    ny1             i     Number of points
 *    zv1             i     z vector np points
 *    nz1             i     Number of points
 *    np1             i     Number of points
 *    xv2             i     x vector np points
 *    nx2             i     Number of points
 *    yv2             i     y vector np points
 *    ny2             i     Number of points
 *    zv2             i     z vector np points
 *    nz2             i     Number of points
 *    xtol, ... itol  i     Tolerances for X Y Z, INCL
 *    option          i     Options: for future usage
 *
 * RETURNS:
 *    Function:  0: Upon success. If problems:
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    CF. XTG license
 ******************************************************************************
 */


#include "libxtg.h"
#include "libxtg_.h"

int well_trunc_parallel(
                        double *xv1,
                        long nx1,
                        double *yv1,
                        long ny1,
                        double *zv1,
                        long nz1,
                        double *xv2,
                        long nx2,
                        double *yv2,
                        long ny2,
                        double *zv2,
                        long nz2,
                        double xtol,
                        double ytol,
                        double ztol,
                        double itol,
                        double atol,
                        int option
                        )
{
    /* locals */
    int i1, i2, ier1, ier2;
    double *md1=NULL, *md2=NULL;
    double *in1=NULL, *in2=NULL;
    double *az1=NULL, *az2=NULL;

    md1 = calloc(nx1, sizeof(double));
    in1 = calloc(nx1, sizeof(double));
    az1 = calloc(nx1, sizeof(double));
    md2 = calloc(nx2, sizeof(double));
    in2 = calloc(nx2, sizeof(double));
    az2 = calloc(nx2, sizeof(double));

    /* first compute inclinations */
    ier1 = well_geometrics(nx1, xv1, yv1, zv1, md1, in1, az1, 0);
    ier2 = well_geometrics(nx2, xv2, yv2, zv2, md2, in2, az2, 0);

    if (ier1 != 0 || ier2 != 0) {
        logger_error(__LINE__, "Something went wrong on well geometrics in %s", __FUNCTION__);
        return EXIT_FAILURE;
    }

    /* compare point by point and update 1 etc if close to a point in 2 */
    for (i1 = 0; i1 < nx1; i1++) {
        for (i2 = 0; i2 < nx2; i2++) {
            if (fabs(xv1[i1] - xv2[i2]) > xtol) continue;
            if (fabs(yv1[i1] - yv2[i2]) > ytol) continue;
            if (fabs(zv1[i1] - zv2[i2]) > ztol) continue;
            if (fabs(x_diff_angle(in1[i1], in2[i2], 1, 0)) > itol) continue;
            if (fabs(x_diff_angle(az1[i1], az2[i2], 1, 0)) > atol) continue;

            xv1[i1] = UNDEF;
            yv1[i1] = UNDEF;
            zv1[i1] = UNDEF;
        }
    }

    free(md1);
    free(md2);
    free(in1);
    free(in2);
    free(az1);
    free(az2);

    return EXIT_SUCCESS;
}
