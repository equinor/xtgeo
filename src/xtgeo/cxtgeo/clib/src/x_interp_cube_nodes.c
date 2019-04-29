/*
 ******************************************************************************
 *
 * NAME:
 *    x_interp_cube_node.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    This routine finds the interpolation within a set of 8 cube nodes
 *    The routine assumes that the point is inside the nodes given by x_v,
 *    y_v, and z_v. This must be checked by the calling routine, but a
 *    test here is also provided.
 *    This works in cube grid where XY can be rotated, given that
 *    that point is given as relative coordinates first.
 *    No rotation in Z (i.e. Z plane is fully horizontal)
 *
 * The points should be organized as follows (nonrotated cube):
 *
 *     2       3          N
 *                        |            TOP
 *     0       1          |___E
 *
 *
 *     6       7          N
 *                        |            BASE
 *     4       5          |___E
 *
 * ARGUMENTS:
 *    x_v, y_v, z_v, p_v     i     Coordinates for XYZ, 8 corners + values
 *                                 in a NONROTATED coordinate system
 *    x, y, z                i     The RELATIVE point to interpolate to
 *    value                  o     Computed value (output)
 *    method                 i     1: Trilinear interpolation (general)
 *    debug                  i     Debug level
 *
 * RETURNS:
 *     0:  if success, otherwise...
 *    -1:  point is outside cell
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ******************************************************************************
 */

#include <math.h>
#include "libxtg.h"
#include "libxtg_.h"


int x_interp_cube_nodes (
                         double *x_v,
                         double *y_v,
                         double *z_v,
                         float *p_v,
                         double x,
                         double y,
                         double z,
                         float *value,
                         int method,
                         int debug
                         )
{
    /* locals */
    char s[24] = "x_interp_cube_nodes";
    double len1, len2, len3, tlen1, tlen2, tlen3, vtot, vsub,
        w[8], sumw = 0.0, vv = 0.0;
    int i, flagundef = 0;

    xtgverbose(debug);
    if (debug > 2) xtg_speak(s, 3, "Entering %s", s);

    /*
     * ########################################################################
     * Method 1 use cubic interpolation (cf wikipedia; in particular the
     * figure: https://en.wikipedia.org/wiki/Trilinear_interpolation#/media/
     * File:Trilinear_interpolation_visualisation.svg)
     * ########################################################################
     */
    if (method==1) {

        flagundef = 0;  /* to take dead traces with UNDEF value into account */

	/*
	 * OK, we are inside mapnodes. Now I need the weights (in 3D)
	 * |---------------|   To get the wight of point Q, need to compute
	 * |   qw      |   |   the opposite volume qw (3D) and divide by total
	 * |-----------*   |   total valume. This is done for ALL corners
	 * |               |
	 * |               |
	 * |               |
	 * |---------------Q
	 */

        /* total volume: */
        tlen1 = fabs(x_v[0] - x_v[1]) + FLOATEPS;  // Add FLOATEPS:
        tlen2 = fabs(y_v[0] - y_v[2]) + FLOATEPS;  // avoid numerical artifacts
        tlen3 = fabs(z_v[0] - z_v[4]) + FLOATEPS;

        vtot = tlen1 * tlen2 * tlen3;

        if (debug > 2) xtg_speak(s,3,"Vtot is %f (%f %f %f)",
                                 vtot, tlen1, tlen2, tlen3);

        if (debug > 2) xtg_speak(s,3,"Relative point (%f %f %f)",
                                 x, y, z);


        /* corner 0 has the opposite corner 7 etc */
        sumw = 0.0;
        vv = 0.0;
        for (i = 0; i < 8; i++) {
            len1 = fabs(x_v[7-i] - x);
            len2 = fabs(y_v[7-i] - y);
            len3 = fabs(z_v[7-i] - z);

            if (debug > 2) xtg_speak(s, 3, "LEN 1 2 3 %f %f %f",
                                     len1, len2, len3);

            if (len1 > tlen1 || len2 > tlen2 || len3 > tlen3) {
                xtg_warn(s, 2, "Point outside, skip");
                return -1;
            }

            vsub = len1 * len2 * len3;
            w[i] = vsub / vtot;

            if (p_v[i] > UNDEF_LIMIT) {
                flagundef = 1;
            }
            else if (p_v[i] < UNDEF_LIMIT) {
                vv = vv + p_v[i] * w[i];
                sumw += w[i];
            }

            if (debug > 2) xtg_speak(s, 3, "Corner %d: %lf %lf %lf",
                                     i, x_v[i], y_v[i], z_v[i]);

            if (debug > 2) xtg_speak(s, 3, "Input value + weigth %lf %lf",
                                     p_v[i], w[i]);

        }

        if (flagundef == 0 && fabs(sumw - 1.0) > 5.0 * FLOATEPS) {
            xtg_warn(s, 1, "Sum of weight not approx equal 1: %lf", sumw);
            return (-5);
        }

        if (flagundef == 1) {
            if (sumw > FLOATEPS) {
                vv = vv * 1.0/sumw;  /* scale weights */
            }
            else{
                vv = UNDEF;
            }
        }
    }

    *value = vv;

    return EXIT_SUCCESS;
}
