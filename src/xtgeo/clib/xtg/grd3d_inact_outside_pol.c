
/*
 ***************************************************************************************
 *
 * NAME:
 *    grd3d_inact_outs_pol.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Check all cells in a layer or subgrid, and set active to zero if outside
 *    or inside
 *
 *    The algorithm is to see if grid nodes lies inside some of the polygons.
 *    If not, an undef value is given. If already undef, then value is kept.
 *
 *    Note, polygons must have a 999 flag line for each new polygon
 *
 * ARGUMENTS:
 *    p_xp_v ...     i     Polygons vectors
 *    npx, npy       i     number of points in the polygon
 *    nx, ny, nz     i     Dimensions for 3D grid
 *    coord.. zcorn  i     Grid coordinates
 *    actnum        i/o    ACTNUM array
 *    nn             i     Length of array (for SWIG)
 *    k1, k2         i     K range min max
 *    option         i     0 inactivate inside, 1 inactivate outside
 *    debug          i     Debug level
 *
 * RETURNS:
 *    0 if all is OK, 1 if there are polygons that have problems.
 *    Result ACNUM is updated
 *
 * TODO/ISSUES/BUGS:
 *    Todo: The algorithm is straightforward and hence a bit slow...
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 **************************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"
#include "logger.h"

/* skip subgrids; use K ranges, multiple polygons allowed */

int
grd3d_inact_outside_pol(double *p_xp_v,
                        long npx,
                        double *p_yp_v,
                        long npy,
                        int nx,
                        int ny,
                        int nz,
                        double *coordsv,
                        long ncoordin,
                        double *zcornsv,
                        long nzcornin,
                        int *p_actnum_v,
                        long nact,
                        int k1,
                        int k2,
                        int force_close,
                        int option)
{
    int i, j, k, ic, istat, np1, np2, ier = 0;
    double xg, yg, zg;
    int iflag, npoly;

    if (option == 0) {
        logger_info(LI, FI, FU, "Masking a grid with polygon (UNDEF outside) ...");
    } else {
        logger_info(LI, FI, FU, "Masking a grid with polygon (UNDEF inside) ...");
    }

    for (k = k1; k <= k2; k++) {

        for (j = 1; j <= ny; j++) {
            for (i = 1; i <= nx; i++) {
                grd3d_midpoint(i, j, k, nx, ny, nz, coordsv, zcornsv, &xg, &yg, &zg,
                               XTGDEBUG);

                long ib = x_ijk2ib(i, j, k, nx, ny, nz, 0);

                /* check all polygons */
                /* for outside, need to make flag system so the cell is */
                /* outside all polys that are spesified */

                iflag = 0;

                np1 = 0;
                npoly = 0;

                for (ic = 0; ic < npx; ic++) {
                    if (p_xp_v[ic] == 999.0) {
                        np2 = ic - 1;
                        if (np2 > np1 + 2) {

                            istat = polys_chk_point_inside(xg, yg, p_xp_v, p_yp_v, np1,
                                                           np2, XTGDEBUG);

                            if (istat < 0) {
                                /* some problems, .eg. poly is not closed */
                                ier = 1;
                            } else {
                                if (option == 0 && istat > 0) {
                                    iflag = 1;
                                }

                                else if (option == 1 && istat == 0) {
                                    iflag++;
                                }
                                npoly++;
                            }
                        }
                        np1 = ic + 1;
                    }
                }

                if (option == 0 && iflag == 1) {
                    p_actnum_v[ib] = 0;
                }
                if (option == 1 && iflag > 0 && iflag == npoly) {
                    p_actnum_v[ib] = 0;
                }
            }
        }
    }
    logger_info(LI, FI, FU, "Masking a grid with polygon... done");
    return ier;
}
