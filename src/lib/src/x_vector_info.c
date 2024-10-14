/*
 ***************************************************************************************
 *
 * NAME:
 *    x_vector_info2
 *
 * DESCRIPTION:
 *    Take 2 points in XY space and compute length and azimuth or normal angle
 *    Angles shall be in range 0-360 degrees (no negative angles)
 *
 * ARGUMENTS:
 *    x1 ... y2        i     Points
 *    vlen             o     Length (2D, XY space)
 *    xangle_radian    o     Angle, radians
 *    xangle_degrees   o     Angle, degrees
 *    option           i     -1: No angle computation (increase speed)
 *                            0: azimuth returned,
 *                            1: angle (aka school) is returned
 *
 * RETURNS by pointers:
 *    Option -1, Lengths only
 *    Option  0, Lengths + AZIMUTH is returned (clockwise, releative to North)
 *    Option  1, Lengths + ANGLE is returned (counter clockwise, relative to East)
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    CF XTGeo's LICENSE
 ***************************************************************************************
 */
#include <math.h>
#include <stdlib.h>
#include <xtgeo/xtgeo.h>
#include "common.h"
#include "logger.h"

void
x_vector_info2(double x1,
               double x2,
               double y1,
               double y2,
               double *vlen,
               double *xangle_radian,
               double *xangle_degrees,
               int option)
{
    /* locals */
    double azi, deg;

    /*
     * ------------------------------------------------------------------------
     * Some checks
     * ------------------------------------------------------------------------
     */

    if (x1 == x2 && y1 == y2) {
        *vlen = 0.000001;
        *xangle_radian = 0.0;
        *xangle_degrees = 0.0;
        return;
    }

    /*
     * ------------------------------------------------------------------------
     * Compute
     * ------------------------------------------------------------------------
     */

    *vlen = sqrt(powf(x2 - x1, 2) + powf(y2 - y1, 2));

    if (option == -1)
        return;

    if ((x2 - x1) > 0.00001 || (x2 - x1) < -0.00001) {

        deg = atan((y2 - y1) / (x2 - x1));
        /* western quadrant */
        if (x2 > x1) {
            azi = PI / 2 - deg;
        }
        /* eastern quadrant */
        else {
            deg = deg + PI;
            azi = 2 * PI + PI / 2 - deg;
        }

    } else {
        if (y2 < y1) {
            azi = PI;
            deg = -PI / 2.0;
        } else {
            azi = 0;
            deg = PI / 2;
        }
    }

    if (azi < 0)
        azi = azi + 2 * PI;
    if (azi > 2 * PI)
        azi = azi - 2 * PI;

    if (deg < 0)
        deg = deg + 2 * PI;
    if (deg > 2 * PI)
        deg = deg - 2 * PI;

    *xangle_radian = azi;

    /* normal school angle */
    if (option == 1) {
        *xangle_radian = deg;
    }

    *xangle_degrees = *(xangle_radian) * 180 / PI;
}

/*
 ***************************************************************************************
 *
 * NAME:
 *    x_vector_len3d*
 *
 * DESCRIPTION:
 *    Length of line in 3D between P1 and P2
 *
 ***************************************************************************************
 */

double
x_vector_len3d(double x1, double x2, double y1, double y2, double z1, double z2)
{
    double vlen;

    if (x1 == x2 && y1 == y2 && z1 == z2) {
        vlen = 10E-20;
    } else {
        vlen = sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2) + pow(z2 - z1, 2));
    }
    return vlen;
}

double
x_vector_len3dx(double x1, double y1, double z1, double x2, double y2, double z2)
// aa but different order of input items
{
    double vlen;

    if (x1 == x2 && y1 == y2 && z1 == z2) {
        vlen = 10E-20;
    } else {
        vlen = sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2) + pow(z2 - z1, 2));
    }
    return vlen;
}

/*
 ***************************************************************************************
 *
 * NAME:
 *    x_vectorpair_angle3d
 *
 * DESCRIPTION:
 *    Find the positive angle a between in 2 lines in space
 *
 *    p1
 *     \
 *      \                      cos(a) = A dot B / |A| |B|
 *   A   \ a
 *        \------------ p2
 *        p0     B
 *
 *
 * Input arguments have lengths for numpy SWIG usage (shall be 3)
 * degrees = 1: return result in degrees
 * option = 1: set all Z values to 0 (get bird perspective angles)
 *
 * Return: Angle in radians or degrees
 *         -888 if wrong dimensions
 *         -999 if at least one vector is too short to determine (cell collapse)
 ***************************************************************************************
 */

double
x_vectorpair_angle3d(double *p0,
                     long n0,
                     double *p1,
                     long n1,
                     double *p2,
                     long n2,
                     int degrees,
                     int option)
{

    if (n0 != n1 || n0 != n2 || n0 != 3) {
        return -888;  // shall ~never happen
    }

    // directional vectors
    double vax = p1[0] - p0[0];
    double vay = p1[1] - p0[1];
    double vaz = p1[2] - p0[2];

    double vbx = p2[0] - p0[0];
    double vby = p2[1] - p0[1];
    double vbz = p2[2] - p0[2];

    if (option == 1) {
        vaz = 0.0;
        vbz = 0.0;
    }

    // dot product
    double dotp = vax * vbx + vay * vby + vaz * vbz;

    // magnitudes
    double magva = sqrt(vax * vax + vay * vay + vaz * vaz);
    double magvb = sqrt(vbx * vbx + vby * vby + vbz * vbz);

    if (magva < FLOATEPS || magvb < FLOATEPS)
        return -999;

    double cosangle = dotp / (magva * magvb);

    double angle = acos(cosangle);

    if (degrees == 1)
        angle = angle * 180 / M_PI;

    return angle;
}

/*
 ***************************************************************************************
 *
 * NAME:
 *    x_minmax_cellangles_topbase
 *
 * DESCRIPTION:
 *    Find all angles a* between in 4 points in space that makes a cell top or a
 *    cell base 3D with common origin (typical grid cells organisation) and return
 *    min and max values
 *
 *  (6,7,8)                (9,10,11)
 *    p2
 *     \------------------/ p3
 *      \a2           a3 /
 *       \ a0       a1  /
 *        \------------/
 *        p0          p1
 *      (0,1,2)       (3,4,5)
 *
   ARGUMENTS:
 *    pvec             i     Vector with 8 corners, as (x0, y0, z0, x1, y1, ... z7)
 *                           cf cell vectors in e.g. grdcp3d_corners.c
 *    a0, a1, ..       o     Output angles (single pointers)
 *    option           i     0: top cell, 1: base cell, 10 top cell projected, 11 base
 *                           cell projected
 *    degrees          i     1 if result in degrees
 ***************************************************************************************
 */

static void
_slarr(const double *array, double *point, int start, int end)
{
    // getting av slice of the 24 elem long array based on indices

    int i;
    int j = 0;
    for (i = start; i <= end; i++) {
        point[j++] = array[i];
    }
}

int
x_minmax_cellangles_topbase(double *pvec,
                            long nvec,
                            double *amin,
                            double *amax,
                            int option,
                            int degrees)
{
    if (nvec != 24) {
        return -9;  // shall ~never happen
    }

    int bird = 0;
    if (option == 1)
        bird = 1;

    // this is a list to first X index position in pvec and the two next are Y Z, e.g.
    // 0 means pvec[0, 1, 2] (x y z of first cell corner), so {0, 3, 6} points to
    // p0, p1, p2 ==> a0, while {9, 6, 3} means p3, p2, p1 ==> a3

    //             --a0--   --a1--   --a2--   --a3--
    int ind[] = { 0, 3, 6, 3, 9, 0, 6, 0, 9, 9, 6, 3 };

    *amin = VERYLARGEPOSITIVE;
    *amax = VERYLARGENEGATIVE;

    int side;
    for (side = 0; side < 2; side++) {

        int cadd = 0;
        if (side == 1)
            cadd = 12;

        int dobreak = 0;
        int i;
        for (i = 0; i < 12; i += 3) {

            double *p0 = calloc(3, sizeof(double));
            double *p1 = calloc(3, sizeof(double));
            double *p2 = calloc(3, sizeof(double));

            _slarr(pvec, p0, ind[i + 0] + cadd, ind[i + 0] + cadd + 2);
            _slarr(pvec, p1, ind[i + 1] + cadd, ind[i + 1] + cadd + 2);
            _slarr(pvec, p2, ind[i + 2] + cadd, ind[i + 2] + cadd + 2);

            double angle = x_vectorpair_angle3d(p0, 3, p1, 3, p2, 3, degrees, bird);

            if (angle > *amax)
                *amax = angle;
            if (angle < *amin)
                *amin = angle;

            // one vector is zero -> collapse
            if (angle == -999) {
                *amin = 0.0;
                *amax = 180.0;
                dobreak = 1;
            }

            free(p0);
            free(p1);
            free(p2);
        }
        if (dobreak == 1)
            break;
    }
    return EXIT_SUCCESS;
}

/*
 ***************************************************************************************
 *
 * NAME:
 *    x_minmax_cellangles_sides
 *
 * DESCRIPTION:
 *    Find all angles a* between in 4 cell sides and return min and max values
 *
 *  (0,1,2)                (3,4,5)
 *    p2       FRONT
 *     \------------------/ p3
 *      \a2           a3 /
 *       \ a0       a1  /
 *        \------------/
 *        p0          p1
 *      (12,13,14)       (15,16,17)
 *
 *  (3,4,5)                (9,10,11)
 *    p2       RIGHT
 *     \------------------/ p3
 *      \a2           a3 /
 *       \ a0       a1  /
 *        \------------/
 *        p0          p1
 *      (15,16,17)       (21,22,23)
 *
 *  (6,7,8)                (0,1,2)
 *    p2       LEFT
 *     \------------------/ p3
 *      \a2           a3 /
 *       \ a0       a1  /
 *        \------------/
 *        p0          p1
 *      (18,19,20)       (12,13,14)
 *
 *  (9,10,11)                (6,7,8)
 *    p2       BACK
 *     \------------------/ p3
 *      \a2           a3 /
 *       \ a0       a1  /
 *        \------------/
 *        p0          p1
 *      (21,22,23)       (18,19,20)
 *
 * ARGUMENTS:
 *    pvec             i     Vector with 8 corners, as (x0, y0, z0, x1, y1, ... z7)
 *                           cf cell vectors in e.g. grdcp3d_corners.c
 *    a0, a1, ..       o     Output angles (single pointers)
 *    degrees          i     1 if result in degrees
 ***************************************************************************************
 */

int
x_minmax_cellangles_sides(double *pvec,
                          long nvec,
                          double *amin,
                          double *amax,
                          int degrees)
{
    if (nvec != 24) {
        return -9;  // shall ~never happen
    }

    // this is a list to first X index position in pvec and the two next are Y Z, e.g.
    // 0 means pvec[0, 1, 2] (x y z of first cell corner), so {12, 15, 0} points to
    // p0, p1, p2 ==> a0 for FRONT case...

    //                   --a0--       --a1--     --a2--   --a3--
    int ind[4][12] = { { 12, 15, 0, 15, 3, 12, 0, 12, 3, 3, 0, 15 },    // FRONT
                       { 15, 21, 3, 21, 9, 15, 3, 15, 9, 9, 3, 21 },    // RIGHT
                       { 18, 12, 6, 12, 0, 18, 6, 18, 0, 0, 6, 12 },    // LEFT
                       { 21, 18, 9, 18, 6, 21, 9, 21, 6, 6, 9, 18 } };  // BACK

    *amin = VERYLARGEPOSITIVE;
    *amax = VERYLARGENEGATIVE;

    int side;
    for (side = 0; side < 4; side++) {

        int dobreak = 0;
        int i;
        for (i = 0; i < 12; i += 3) {

            double *p0 = calloc(3, sizeof(double));
            double *p1 = calloc(3, sizeof(double));
            double *p2 = calloc(3, sizeof(double));

            _slarr(pvec, p0, ind[side][i + 0], ind[side][i + 0] + 2);
            _slarr(pvec, p1, ind[side][i + 1], ind[side][i + 1] + 2);
            _slarr(pvec, p2, ind[side][i + 2], ind[side][i + 2] + 2);

            double angle = x_vectorpair_angle3d(p0, 3, p1, 3, p2, 3, degrees, 0);

            if (angle > *amax)
                *amax = angle;
            if (angle < *amin)
                *amin = angle;

            // one vector is zero -> collapse
            if (angle == -999) {
                *amin = 0.0;
                *amax = 180.0;
                dobreak = 1;
            }

            free(p0);
            free(p1);
            free(p2);
        }
        if (dobreak == 1)
            break;
    }

    return EXIT_SUCCESS;
}
