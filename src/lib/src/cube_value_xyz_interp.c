/*
 ***************************************************************************************
 *
 * NAME:
 *    cube_value_xyz_interp.c
 *
 * DESCRIPTION:
 *    Given X Y Z, return interpolated cube value. The XYZ value is
 *    interpolated by trilinear interpolation, i.e. 8 cell values are applied
 *    to estimate the value
 *
 * ARGUMENTS:
 *    x, y, z        i     Position in cube to request a value
 *    xinc.. yflip   i     Cube geometry settings
 *    nx ny nz       i     Cube dimensions
 *    p_val_v        i     3D cube values
 *    value          o     Updated value valid for X Y Z postion
 *    option         i     If 1: snap to nearest cube node in X Y
 *                   i     If >= 10: Skip I J calculation
 *
 * RETURNS:
 *    Function:  0: upon success.
 *              -1: XYZ given is outside cube
 *              -2: Something went wrong in trilinar interpolation
 *    Result value is updated
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ***************************************************************************************
 */
#include <math.h>
#include <stdlib.h>
#include <xtgeo/xtgeo.h>
#include "common.h"
#include "logger.h"

int
cube_value_xyz_interp(double xin,
                      double yin,
                      double zin,
                      double xori,
                      double xinc,
                      double yori,
                      double yinc,
                      double zori,
                      double zinc,
                      double rot_deg,
                      int yflip,
                      int nx,
                      int ny,
                      int nz,
                      float *p_val_v,
                      float *value,
                      int option)
{
    /* locals */
    long ib;
    int ic, jc, kc, i, j, k, ier, ier1, flag;
    double x_v[8], y_v[8], z_v[8], xx, yy, zz, rx, ry, rz;
    float p_v[8], val;

    /* need to determine the lower left corner coordinates of the point ie
       need to run with flag = 1 */
    flag = 1;
    if (option >= 10)
        flag = 11;

    ier = cube_ijk_from_xyz(&ic, &jc, &kc, &rx, &ry, &rz, xin, yin, zin, xori, xinc,
                            yori, yinc, zori, zinc, nx, ny, nz, rot_deg, yflip, flag);

    if (ier == -1) {
        *value = UNDEF;
        return -1;
    }

    /* make possibility to snap to nearest cube corner (auto4d request) */
    if (option == 1 || option == 11) {
        ib = 0;
        for (k = 0; k <= 1; k++) {
            for (j = 0; j <= 1; j++) {
                for (i = 0; i <= 1; i++) {
                    ier = cube_coord_val_ijk(ic + i, jc + j, kc + k, nx, ny, nz, xori,
                                             xinc, yori, yinc, zori, zinc, rot_deg,
                                             yflip, p_val_v, &xx, &yy, &zz, &val, 0);
                    if (ier == 0) {
                        x_v[ib] = xx;
                        y_v[ib] = yy;
                        z_v[ib] = zz;
                        ib++;
                    }
                }
            }
        }
        long ibmax = ib;
        /* determine closest corner: */
        long useib = 0;
        double previousdist = 10E20;
        for (ib = 0; ib < ibmax; ib++) {
            /* find horizontal distance; hence z1 and z2 are both zin */
            double dist = x_vector_len3d(x_v[ib], xin, y_v[ib], yin, zin, zin);
            if (dist < previousdist) {
                useib = ib;
                previousdist = dist;
            }
        }
        double usex = x_v[useib];
        double usey = y_v[useib];

        double avginc = 0.5 * (xinc + yinc);
        if (fabs(previousdist) > fabs(0.1 * avginc)) {
            /* logger_warn("Warning, snapping distance is more than " */
            /*             "10 percent of avg cell size in XY: %f vs %f (%s). " */
            /*             "Consider to deactivate snapxy option?", */
            /*             previousdist, avginc, FU); */
        }

        ier =
          cube_ijk_from_xyz(&ic, &jc, &kc, &rx, &ry, &rz, usex, usey, zin, xori, xinc,
                            yori, yinc, zori, zinc, nx, ny, nz, rot_deg, yflip, flag);

        if (ier == -1) {
            *value = UNDEF;
            return -1;
        }
    }

    /* need to get coordinates and values from all 8 corner values */
    ib = 0;

    /* relative ccords applied! */
    xori = 0.0;
    yori = 0.0;
    zori = 0.0;
    rot_deg = 0.0;

    ier1 = 0;

    /* Note boundaries, and K as outer is intensional
       (also with numpy in C order */

    for (k = 0; k <= 1; k++) {
        for (j = 0; j <= 1; j++) {
            for (i = 0; i <= 1; i++) {

                ier = cube_coord_val_ijk(ic + i, jc + j, kc + k, nx, ny, nz, xori, xinc,
                                         yori, yinc, zori, zinc, rot_deg, yflip,
                                         p_val_v, &xx, &yy, &zz, &val, flag);

                if (ier == 0) {
                    x_v[ib] = xx;
                    y_v[ib] = yy;
                    z_v[ib] = zz;
                    p_v[ib] = val;
                } else {
                    ier1 = ier;
                }

                ib++;
            }
        }
    }

    /* value is outside cube */
    if (ier1 == -1) {
        *value = UNDEF;
        return EXIT_SUCCESS;
    }

    /* now interpolate */
    ier = -9;
    if (ier1 == 0) {
        ier = x_interp_cube_nodes(x_v, y_v, z_v, p_v, rx, ry, rz, &val, 1);
    }

    if (ier != 0) {
        *value = UNDEF;
        return (ier);
    } else {
        *value = val;
    }

    return EXIT_SUCCESS;
}
