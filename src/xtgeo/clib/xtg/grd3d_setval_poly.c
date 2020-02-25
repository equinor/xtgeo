

/*
 ******************************************************************************
 *
 * NAME:
 *    grd3d_setval_poly.c
 *
 * DESCRIPTION:
 *    Check all cells and set value if inside polygon for all
 *    layers. This can be used to make a "proxy" for numpy operations
 *
 * ARGUMENTS:
 *    p_xp_v ...     i     Polygons vector
 *    npx, npy       i     Number of points in the polygon
 *    nx, ny, nz     i     Dimensions for 3D grid
 *    coord.. zcorn  i     Grid coordinates
 *    actnum         i     ACTNUM array
 *    p_prop_v      i/o    Property array (to modify)
 *    value          i     Value to use
 *
 * RETURNS:
 *    0 if all is OK, 1 if there are polygons that have problems.
 *    Resulting property is updated.
 *    Return number:
 *    -9 Polygon is not closed
 *
 * LICENCE:
 *    CF. XTGeo license
 ******************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"
#include "logger.h"


/* the Python version; simple as possible, (more operations in numpy)*/
int
grd3d_setval_poly(double *p_xp_v,
                  long npx,
                  double *p_yp_v,
                  long npy,

                  int nx,
                  int ny,
                  int nz,

                  double *p_coord_v,
                  long ncoordin,
                  double *p_zcorn_v,
                  long nzcornin,
                  int *p_actnum_v,
                  long nactin,

                  double *p_prop_v,
                  double value)
{
    int i, j, k, istat = 0;
    long ib;
    double xg, yg, zg;

    logger_info(LI, FI, FU, "Set proxy value wrt polygon...");

    for (k = 1; k <= nz; k++) {

        for (j = 1; j <= ny; j++) {

            for (i = 1; i <= nx; i++) {

                grd3d_midpoint(i, j, k, nx, ny, nz, p_coord_v, p_zcorn_v, &xg, &yg, &zg,
                               XTGDEBUG);

                ib = x_ijk2ib(i, j, k, nx, ny, nz, 0);

                /* check polygon */
                istat = pol_chk_point_inside(xg, yg, p_xp_v, p_yp_v, npx, XTGDEBUG);

                if (istat == -9)
                    return istat; /* poly is not closed */

                if (istat > 0 && p_actnum_v[ib] == 1)
                    p_prop_v[ib] = value;
            }
        }
    }

    logger_info(LI, FI, FU, "Set proxy value wrt polygon... done");

    return EXIT_SUCCESS;
}
