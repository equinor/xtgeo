/*
****************************************************************************************
*
* NAME:
*    well_surf_picks.c
*
* DESCRIPTION:
*    Given a trajectory with X Y Z (MD) coordinates, find the intersection with
*    a surface. Also MD and directions are provided. For directions, 1 means
*    well is crossing from top, while -1 means from base
*
* ARGUMENTS:
*    xv, yv, zv     i     x y z vector with dimensions for swig
*    mdv            i     mdepth vector with dimension for swig, note may have UNDEF!
*    ncol .. rot    i     Metadata for surface
*    surfv          i     RegularSurface vector with swig dimension
*    xoutv .. zoutv o     output vectors X Y Z with dimensions
*    mdoutv, doutv  o     output vectors for MD and directions.
*
* RETURNS:
*    Function:  0: Number of points found. Updated arrays
*
* TODO/ISSUES/BUGS:
*
* LICENCE:
*    cf. XTGeo LICENSE
****************************************************************************************
*/

#include "libxtg.h"
#include "libxtg_.h"
#include "logger.h"

int
well_surf_picks(double *xv,
                long nxv,
                double *yv,
                long nyv,
                double *zv,
                long nzv,
                double *mdv,
                long nmdv,

                int ncol,
                int nrow,
                double xori,
                double yori,
                double xinc,
                double yinc,
                int yflip,
                double rot,
                double *surfv,
                long nsurf,

                double *xoutv,
                long nxoutv,
                double *youtv,
                long nyoutv,
                double *zoutv,
                long nzoutv,
                double *mdoutv,
                long nmdoutv,
                int *doutv,
                long ndoutv)
{

    logger_info(LI, FI, FU, "Finding picks, intersections well surface: %s", FU);

    double *zd;

    zd = calloc(nzv, sizeof(double));

    surf_get_zv_from_xyv(xv, nxv, yv, nyv, zd, nzv, ncol, nrow, xori, yori, xinc, yinc,
                         yflip, rot, surfv, nsurf);

    /* compute diff */
    int i;
    for (i = 0; i < nzv; i++) {
        if (zd[i] < UNDEF_LIMIT)
            zd[i] = zv[i] - zd[i];
    }

    /* find crossing and do interpolation */
    int ili = 0;
    for (i = 1; i < nzv; i++) {
        double z0 = zd[i - 1];
        double z2 = zd[i];

        if (z0 > UNDEF_LIMIT || z2 > UNDEF_LIMIT)
            continue;

        if (z0 <= 0 && z2 > 0) {
            xoutv[ili] = x_vector_linint3(z0, 0.0, z2, xv[i - 1], xv[i]);
            youtv[ili] = x_vector_linint3(z0, 0.0, z2, yv[i - 1], yv[i]);
            zoutv[ili] = x_vector_linint3(z0, 0.0, z2, zv[i - 1], zv[i]);
            doutv[ili] = 1;
            mdoutv[ili] = UNDEF;
            if (mdv[i - 1] < UNDEF_LIMIT && mdv[i] < UNDEF_LIMIT)
                mdoutv[ili] = x_vector_linint3(z0, 0.0, z2, mdv[i - 1], mdv[i]);
            logger_debug(LI, FI, FU, "Point found %d %lf", ili, zoutv[ili]);
            ili++;
        }
        if (z0 >= 0 && z2 < 0) {
            xoutv[ili] = x_vector_linint3(z2, 0.0, z0, xv[i], xv[i - 1]);
            youtv[ili] = x_vector_linint3(z2, 0.0, z0, yv[i], yv[i - 1]);
            zoutv[ili] = x_vector_linint3(z2, 0.0, z0, zv[i], zv[i - 1]);
            doutv[ili] = -1;
            mdoutv[ili] = UNDEF;
            if (mdv[i - 1] < UNDEF_LIMIT && mdv[i] < UNDEF_LIMIT)
                mdoutv[ili] = x_vector_linint3(z2, 0.0, z0, mdv[i], mdv[i - 1]);
            logger_debug(LI, FI, FU, "Point found %d %lf", ili, zoutv[ili]);
            ili++;
        }
    }

    logger_info(LI, FI, FU, "Finding picks, intersections well surface, done: %s", FU);

    free(zd);
    return ili;
}
