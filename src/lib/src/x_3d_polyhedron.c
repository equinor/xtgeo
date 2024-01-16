#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <xtgeo/xtgeo.h>

#include "common.h"

const int TETRACOMBS[4][6][4] = {
    // cell top/base hinge is splittet 0 - 3 / 4 - 7
    {
      // lower right common vertex 5
      { 3, 7, 4, 5 },
      { 0, 4, 7, 5 },
      { 0, 3, 1, 5 },
      // upper left common vertex 6
      { 0, 4, 7, 6 },
      { 3, 7, 4, 6 },
      { 0, 3, 2, 6 },
    },

    // cell top/base hinge is splittet 1 -2 / 5- 6
    {
      // upper right common vertex 7
      { 1, 5, 6, 7 },
      { 2, 6, 5, 7 },
      { 1, 2, 3, 7 },
      // lower left common vertex 4
      { 1, 5, 6, 4 },
      { 2, 6, 5, 4 },
      { 1, 2, 0, 4 },
    },

    // Another combination...
    // cell top/base hinge is splittet 0 - 3 / 4 - 7
    {
      // lower right common vertex 1
      { 3, 7, 0, 1 },
      { 0, 4, 3, 1 },
      { 4, 7, 5, 1 },
      // upper left common vertex 2
      { 0, 4, 3, 2 },
      { 3, 7, 0, 2 },
      { 4, 7, 6, 2 },
    },

    // cell top/base hinge is splittet 1 -2 / 5- 6
    { // upper right common vertex 3
      { 1, 5, 2, 3 },
      { 2, 6, 1, 3 },
      { 5, 6, 7, 3 },
      // lower left common vertex 0
      { 1, 5, 2, 0 },
      { 2, 6, 1, 0 },
      { 5, 6, 4, 0 } }

};

static inline double
x_hexahedron_dz(double *corners)
{
    // TODO: This does not account for overall zflip ala Petrel or cells that
    // are malformed
    double dzsum = 0.0;
    for (int i = 0; i < 4; i++) {
        dzsum += fabs(corners[3 * i + 2] - corners[3 * i + 2 + 12]);
    }
    return dzsum / 4.0;
}

static int
_x_point_outside_hexahedron_simple(double px, double py, double pz, double *corners)
{
    // return 1 if point is definitively outside cell; otherwise _maybe_ inside

    double xmin = VERYLARGEPOSITIVE;
    double ymin = VERYLARGEPOSITIVE;
    double zmin = VERYLARGEPOSITIVE;
    double xmax = VERYLARGENEGATIVE;
    double ymax = VERYLARGENEGATIVE;
    double zmax = VERYLARGENEGATIVE;

    double xarr[8];
    double yarr[8];
    double zarr[8];
    for (int nc = 0; nc < 8; nc++) {
        xarr[nc] = corners[0 + nc * 3];
        yarr[nc] = corners[1 + nc * 3];
        zarr[nc] = corners[2 + nc * 3];
        if (xarr[nc] < xmin)
            xmin = xarr[nc];
        if (xarr[nc] > xmax)
            xmax = xarr[nc];
        if (yarr[nc] < ymin)
            ymin = yarr[nc];
        if (yarr[nc] > ymax)
            ymax = yarr[nc];
        if (zarr[nc] < zmin)
            zmin = zarr[nc];
        if (zarr[nc] > zmax)
            zmax = zarr[nc];
    }
    if (px < xmin || px > xmax)
        return 1;
    if (py < ymin || py > ymax)
        return 1;
    if (pz < zmin || pz > zmax)
        return 1;

    return 0;
}

/*
 ***************************************************************************************
 *
 * NAME:
 *    x_tetrehedron_volume.c
 *
 *
 * DESCRIPTION:
 *    Find the volume of a irregular tetrahedron. Based on
 *    www.geeksforgeeks.org/program-to-find-the-volume-of-an-irregular-tetrahedron/
 *
 * ARGUMENTS:
 *    pv          i     a [12] array with X Y Z of 4 vertices, x1, y1, z1, x2, y2, ...
 *
 * RETURNS:
 *    Volume
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ***************************************************************************************
 */

double
x_tetrahedron_volume(double *pv, long ndim)
{
    // length of each edge
    double a = x_vector_len3dx(pv[0], pv[1], pv[2], pv[3], pv[4], pv[5]);
    double b = x_vector_len3dx(pv[0], pv[1], pv[2], pv[6], pv[7], pv[8]);
    double c = x_vector_len3dx(pv[0], pv[1], pv[2], pv[9], pv[10], pv[11]);
    double d = x_vector_len3dx(pv[3], pv[4], pv[5], pv[6], pv[7], pv[8]);
    double e = x_vector_len3dx(pv[3], pv[4], pv[5], pv[9], pv[10], pv[11]);
    double f = x_vector_len3dx(pv[6], pv[7], pv[8], pv[9], pv[10], pv[11]);

    if (a < FLOATEPS || b < FLOATEPS || c < FLOATEPS || d < FLOATEPS || e < FLOATEPS ||
        f < FLOATEPS)
        return 0.0;

    double ap = pow(a, 2);
    double bp = pow(b, 2);
    double cp = pow(c, 2);
    double dp = pow(d, 2);
    double ep = pow(e, 2);
    double fp = pow(f, 2);

    double vol = 4 * (cp * ap * bp) - cp * pow((ap + bp - dp), 2) -
                 ap * pow((bp + cp - fp), 2) - bp * pow((cp + ap - ep), 2) +
                 (ap + bp - dp) * (bp + cp - fp) * (cp + ap - ep);

    if (fabs(vol) < FLOATEPS)
        return 0.0;

    vol = sqrt(vol) / 12.0;

    return vol;
}

/*
 ***************************************************************************************
 *
 * NAME:
 *    x_point_in_tetrehedron.c
 *
 *
 * DESCRIPTION:
 *    Test if a point is inside a tetrahedron. The method is to compare the volume
 *    of the tetrahedron with the sum of tetrahedrons where one vertex is replaced
 *    with the point. If equal, then point is inside or at boundary
 *
 * ARGUMENTS:
 *    x0, y0, z0    i     Point coords
 *    pv            i     a [12] array with X Y Z of 4 vertices, x1, y1, z1, x2, y2, ...
 *    ndim          i     Dimension (for Python SWIG bindings)
 *
 * RETURNS:
 *    100 if 100% inside, otherwise 0:
 *
 *    IN PREP:
 *    Otherwise a number between from 0 to 99 telling the
 *    "closeness" of being inside. In particular:
 *    If the sumvol >= 2 * truevol, then return zero
 *    Otherwise return a number which is "scaled" ,e.g.:
 *       if truevol is 40 and sumvol is 50:  100 * 0.8 = 80
 *       if truevol is 40 and sumvol is 80:  100 * 0.5 = 50
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ***************************************************************************************
 */

int
x_point_in_tetrahedron(double x0, double y0, double z0, double *pv, long ndim)
{

    double truevol = x_tetrahedron_volume(pv, 12);

    if (truevol < FLOATEPS)
        return 0;

    int i, nv;
    double newpv[12];
    double sumvol;

    sumvol = 0.0;
    for (nv = 0; nv < 4; nv++) {
        for (i = 0; i < ndim; i++) {
            newpv[i] = pv[i];
        }
        newpv[0 + 3 * nv] = x0;
        newpv[1 + 3 * nv] = y0;
        newpv[2 + 3 * nv] = z0;

        double vol = x_tetrahedron_volume(newpv, ndim);

        sumvol += vol;
    }

    double relerror = truevol * 0.001;
    double diff = sumvol - truevol;

    if (diff < -1 * relerror) {
        throw_exception("diff < -1 * relerror in x_point_in_tetrahedron");
        return EXIT_FAILURE;
    } else if (diff > relerror) {
        // LATER: make algorithm more smart to tell "closeness" of point
        // if (sumvol / truevol < 5 && sumvol > 0.0) {
        //     int res = (int)100 * (truevol / sumvol);
        //     // logger_debug(LI, FI, FU, "Sumvol TrueVol %lf %lf Return %d", sumvol,
        //                  truevol, res);
        //     return res;
        // }
        return 0;
    } else {
        return 100;
    }
}

static int
x_point_in_tetrahedron_v2(double x0, double y0, double z0, double *pv, long ndim)
{
    // Simpler version, return 0 if outside and 1 if 0 if inside

    double truevol = x_tetrahedron_volume(pv, 12);

    if (truevol < FLOATEPS)
        return 0;

    int i, nv;
    double newpv[12];
    double sumvol;

    sumvol = 0.0;
    for (nv = 0; nv < 4; nv++) {
        for (i = 0; i < ndim; i++) {
            newpv[i] = pv[i];
        }
        newpv[0 + 3 * nv] = x0;
        newpv[1 + 3 * nv] = y0;
        newpv[2 + 3 * nv] = z0;

        double vol = x_tetrahedron_volume(newpv, ndim);

        sumvol += vol;
    }

    double relerror = truevol * 0.001;
    double diff = sumvol - truevol;

    if (diff > relerror)
        return 0;

    return 1;
}

/*
 ***************************************************************************************
 *
 * NAME:
 *    x_point_in_hexahedron.c
 *
 *
 * DESCRIPTION:
 *    Test if a point is inside a irregular hexahedron. The method is to split into
 *    a number of tetrahedrons and check each of those.
 *
 * ARGUMENTS:
 *    x0, y0, z0    i     Point coords
 *    corners       i     a [24] array with X Y Z of 8 vertices, x1, y1, z1, x2, y2, ...
 *                        arranged as usual for corner point cells
 *    method        i     Different algorithms, 1 or 2
 *
 * RETURNS:
 *    100 if inside, 50 if possibly inside, 0 else (aka percent)
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ***************************************************************************************
 */

/* private, method 1 */
static int
_x_point_in_hexahedron_v1(double x0, double y0, double z0, double *corners, long ndim)
{

    // first avoid cells that collapsed in some way
    if (x_hexahedron_dz(corners) < FLOATEPS) {
        return 0;
    }

    // avoid cells that are definitively outside
    if (_x_point_outside_hexahedron_simple(x0, y0, z0, corners) == 1) {
        return 0;
    }

    double crn[8][3];
    int cset[4][5];

    int i, j;
    int ic = 0;
    for (i = 0; i < 8; i++) {
        for (j = 0; j < 3; j++) {
            crn[i][j] = corners[ic++];
        }
    }

    // the hexahedron consists of 5 tetrahedrons

    cset[0][0] = 0;
    cset[1][0] = 2;
    cset[2][0] = 3;
    cset[3][0] = 6;

    cset[0][1] = 0;
    cset[1][1] = 1;
    cset[2][1] = 3;
    cset[3][1] = 5;

    cset[0][2] = 0;
    cset[1][2] = 4;
    cset[2][2] = 5;
    cset[3][2] = 6;

    cset[0][3] = 3;
    cset[1][3] = 6;
    cset[2][3] = 7;
    cset[3][3] = 5;

    cset[0][4] = 0;
    cset[1][4] = 3;
    cset[2][4] = 5;
    cset[3][4] = 6;

    int icset;
    double thd[12];

    int status1 = 0;
    int set1score = 0;
    for (icset = 0; icset < 5; icset++) {
        ic = 0;
        for (i = 0; i < 4; i++) {
            thd[ic + 0] = crn[cset[i][icset]][0];
            thd[ic + 1] = crn[cset[i][icset]][1];
            thd[ic + 2] = crn[cset[i][icset]][2];
            ic += 3;
        }
        int score = x_point_in_tetrahedron(x0, y0, z0, thd, 12);

        if (score == 100) {
            status1 += 50;
            set1score = 0;
            break;
        } else if (score > 0 && score > set1score) {
            set1score += score;
        }
    }

    if (set1score > 0) {
        status1 = set1score / 2;
    }

    // alternative arrangment of tetrahedrons

    cset[0][0] = 0;
    cset[1][0] = 1;
    cset[2][0] = 2;
    cset[3][0] = 4;

    cset[0][1] = 1;
    cset[1][1] = 2;
    cset[2][1] = 3;
    cset[3][1] = 7;

    cset[0][2] = 4;
    cset[1][2] = 5;
    cset[2][2] = 7;
    cset[3][2] = 1;

    cset[0][3] = 6;
    cset[1][3] = 4;
    cset[2][3] = 7;
    cset[3][3] = 2;

    cset[0][4] = 1;
    cset[1][4] = 2;
    cset[2][4] = 4;
    cset[3][4] = 7;

    int status2 = 0;
    int set2score = 0;
    for (icset = 0; icset < 5; icset++) {
        ic = 0;
        for (i = 0; i < 4; i++) {
            thd[ic + 0] = crn[cset[i][icset]][0];
            thd[ic + 1] = crn[cset[i][icset]][1];
            thd[ic + 2] = crn[cset[i][icset]][2];

            ic += 3;
        }

        int score = x_point_in_tetrahedron(x0, y0, z0, thd, 12);

        if (score == 100) {
            status2 += 50;
            set2score = 0;
            break;
        } else if (score > 0 && score > set2score) {
            set2score += score;
        }
    }

    if (set2score > 0) {
        status2 = set2score / 2;
    }

    int status = status1 + status2;

    return status;
}

/* private, method 2 */
static int
_x_point_in_hexahedron_v2(double x0, double y0, double z0, double *corners, long ndim)
{

    // first avoid cells that collapsed in some way
    if (x_hexahedron_dz(corners) < FLOATEPS) {
        return 0;
    }

    // first avoid cells that are definitive outside
    if (_x_point_outside_hexahedron_simple(x0, y0, z0, corners) == 1) {
        return 0;
    }

    double crn[8][3];

    int i, j;
    int ic = 0;
    for (i = 0; i < 8; i++) {
        for (j = 0; j < 3; j++) {
            crn[i][j] = corners[ic++];
        }
    }

    double thd[12];

    int status = 0;
    int icset, ialt;
    for (ialt = 1; ialt <= 2; ialt++) {
        for (icset = 0; icset < 6; icset++) {
            ic = 0;
            for (i = 0; i < 4; i++) {
                thd[ic + 0] = crn[TETRACOMBS[ialt - 1][icset][i]][0];
                thd[ic + 1] = crn[TETRACOMBS[ialt - 1][icset][i]][1];
                thd[ic + 2] = crn[TETRACOMBS[ialt - 1][icset][i]][2];
                ic += 3;
            }
            int score = x_point_in_tetrahedron_v2(x0, y0, z0, thd, 12);
            if (score == 1) {
                status += 1;
                break;
            }
        }
    }
    return status * 50;
}

/* PUBLIC METHOD */

int
x_point_in_hexahedron(double x0,
                      double y0,
                      double z0,
                      double *corners,
                      long ndim,
                      int method)
{
    if (method == 1) {
        return _x_point_in_hexahedron_v1(x0, y0, z0, corners, ndim);
    } else {
        return _x_point_in_hexahedron_v2(x0, y0, z0, corners, ndim);
    }
}
