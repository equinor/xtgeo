#include "libxtg.h"
#include "libxtg_.h"
#include "logger.h"
#include <math.h>

static double
_x_hexahedron_dz(double *corners)
{
    // TODO: This does not account for overall zflip ala Petrel or cells that
    // are malformed

    int ico;
    double dzsum = 0.0;
    for (ico = 0; ico < 4; ico++) {
        double zcsum = fabs(corners[3 * ico + 2] - corners[3 * ico + 2 + 12]);
        dzsum += zcsum;
    }

    return dzsum / 4.0;
}

static int
_x_hexahedron_collapse(double *corners)
{
    // Detect if cell has collapse corners

    int ico;
    for (ico = 0; ico < 4; ico++) {
        double dzdiff = -1 * (corners[3 * ico + 2] - corners[3 * ico + 2 + 12]);
        if (dzdiff < 0.00001)
            return 1;
    }

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
    double a, b, c, d, e, f;

    // length of each edge
    a = x_vector_len3dx(pv[0], pv[1], pv[2], pv[3], pv[4], pv[5]);
    b = x_vector_len3dx(pv[0], pv[1], pv[2], pv[6], pv[7], pv[8]);
    c = x_vector_len3dx(pv[0], pv[1], pv[2], pv[9], pv[10], pv[11]);
    d = x_vector_len3dx(pv[3], pv[4], pv[5], pv[6], pv[7], pv[8]);
    e = x_vector_len3dx(pv[3], pv[4], pv[5], pv[9], pv[10], pv[11]);
    f = x_vector_len3dx(pv[6], pv[7], pv[8], pv[9], pv[10], pv[11]);

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
 *    x_hexahedron_volume.c
 *
 *
 * DESCRIPTION:
 *    Estimate the volume of a hexahedron i.e. a cornerpoint cell. This is a nonunique
 *    entity, but it is approximated by computing two different ways of top/base
 *    splitting and average those.
 *
 * ARGUMENTS:
 *   corners       i     a [24] array with X Y Z of 8 vertices, x1, y1, z1, x2, y2, ...
 *                        arranged as usual for corner point cells
 *
 * RETURNS:
 *    Volume
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ***************************************************************************************
 */

double
x_hexahedron_volume(double *corners, long ndim)
{

    // first avoid cells that collapsed in some way
    if (_x_hexahedron_dz(corners) < FLOATEPS) {
        return 0.0;
    }
    if (_x_hexahedron_collapse(corners) == 1) {
        printf("Collapse corners are present\n");
    }

    double **crn = x_allocate_2d_double(8, 3);
    int **cset = x_allocate_2d_int(4, 6);

    int i, j;
    int ic = 0;
    for (i = 0; i < 8; i++) {
        for (j = 0; j < 3; j++) {
            crn[i][j] = corners[ic++];
        }
    }

    // the hexahedron consists of 6 tetrahedrons

    cset[0][0] = 0;
    cset[1][0] = 4;
    cset[2][0] = 5;
    cset[3][0] = 7;

    cset[0][1] = 0;
    cset[1][1] = 1;
    cset[2][1] = 5;
    cset[3][1] = 7;

    cset[0][2] = 0;
    cset[1][2] = 1;
    cset[2][2] = 3;
    cset[3][2] = 7;

    cset[0][3] = 2;
    cset[1][3] = 4;
    cset[2][3] = 6;
    cset[3][3] = 7;

    cset[0][4] = 0;
    cset[1][4] = 2;
    cset[2][4] = 4;
    cset[3][4] = 7;

    cset[0][5] = 0;
    cset[1][5] = 2;
    cset[2][5] = 3;
    cset[3][5] = 7;

    double thd[12];

    double vol1 = 0;
    int icset;
    for (icset = 0; icset < 6; icset++) {
        ic = 0;
        for (i = 0; i < 4; i++) {
            thd[ic + 0] = crn[cset[i][icset]][0];
            thd[ic + 1] = crn[cset[i][icset]][1];
            thd[ic + 2] = crn[cset[i][icset]][2];
            ic += 3;
        }
        // printf("ICSET1: %d .. %lf\n", icset, x_tetrahedron_volume(thd, 12));
        vol1 += x_tetrahedron_volume(thd, 12);
    }

    // alternative arrangment of tetrahedrons
    cset[0][0] = 0;
    cset[1][0] = 4;
    cset[2][0] = 5;
    cset[3][0] = 6;

    cset[0][1] = 0;
    cset[1][1] = 1;
    cset[2][1] = 5;
    cset[3][1] = 6;

    cset[0][2] = 0;
    cset[1][2] = 1;
    cset[2][2] = 2;
    cset[3][2] = 6;

    cset[0][3] = 2;
    cset[1][3] = 6;
    cset[2][3] = 5;
    cset[3][3] = 7;

    cset[0][4] = 6;
    cset[1][4] = 1;
    cset[2][4] = 5;
    cset[3][4] = 7;

    cset[0][5] = 1;
    cset[1][5] = 2;
    cset[2][5] = 3;
    cset[3][5] = 7;

    double vol2 = 0;

    for (icset = 0; icset < 6; icset++) {
        ic = 0;
        for (i = 0; i < 4; i++) {
            thd[ic + 0] = crn[cset[i][icset]][0];
            thd[ic + 1] = crn[cset[i][icset]][1];
            thd[ic + 2] = crn[cset[i][icset]][2];
            ic += 3;
        }
        // printf("ICSET2: %d .. %lf\n", icset, x_tetrahedron_volume(thd, 12));
        vol2 += x_tetrahedron_volume(thd, 12);
    }

    x_free_2d_double(crn);
    x_free_2d_int(cset);

    printf("Vol1 and vol2: %lf %lf\n", vol1, vol2);

    return 0.5 * (vol1 + vol2);
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
        logger_critical(LI, FI, FU, "Something is wrong in %s!", FU);
        exit(323);
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
 *
 * RETURNS:
 *    100 if inside, 50 if possibly inside, 0 else (aka percent)
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ***************************************************************************************
 */

int
x_point_in_hexahedron(double x0, double y0, double z0, double *corners, long ndim)
{

    // first avoid cells that collapsed in some way
    if (_x_hexahedron_dz(corners) < FLOATEPS) {
        return 0;
    }

    double **crn = x_allocate_2d_double(8, 3);
    int **cset = x_allocate_2d_int(4, 6);

    int i, j;
    int ic = 0;
    for (i = 0; i < 8; i++) {
        for (j = 0; j < 3; j++) {
            crn[i][j] = corners[ic++];
        }
    }

    // the hexahedron consists of 6 tetrahedrons
    cset[0][0] = 0;
    cset[1][0] = 4;
    cset[2][0] = 5;
    cset[3][0] = 7;

    cset[0][1] = 0;
    cset[1][1] = 1;
    cset[2][1] = 5;
    cset[3][1] = 7;

    cset[0][2] = 0;
    cset[1][2] = 1;
    cset[2][2] = 3;
    cset[3][2] = 7;

    cset[0][3] = 2;
    cset[1][3] = 4;
    cset[2][3] = 6;
    cset[3][3] = 7;

    cset[0][4] = 0;
    cset[1][4] = 2;
    cset[2][4] = 4;
    cset[3][4] = 7;

    cset[0][5] = 0;
    cset[1][5] = 2;
    cset[2][5] = 3;
    cset[3][5] = 7;

    int icset;
    double thd[12];

    int status1 = 0;
    int set1score = 0;
    for (icset = 0; icset < 6; icset++) {
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
    cset[1][0] = 4;
    cset[2][0] = 5;
    cset[3][0] = 6;

    cset[0][1] = 0;
    cset[1][1] = 1;
    cset[2][1] = 5;
    cset[3][1] = 6;

    cset[0][2] = 0;
    cset[1][2] = 1;
    cset[2][2] = 2;
    cset[3][2] = 6;

    cset[0][3] = 2;
    cset[1][3] = 6;
    cset[2][3] = 5;
    cset[3][3] = 7;

    cset[0][4] = 6;
    cset[1][4] = 1;
    cset[2][4] = 5;
    cset[3][4] = 7;

    cset[0][5] = 1;
    cset[1][5] = 2;
    cset[2][5] = 3;
    cset[3][5] = 7;

    int status2 = 0;
    int set2score = 0;
    for (icset = 0; icset < 6; icset++) {
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

    x_free_2d_double(crn);
    x_free_2d_int(cset);

    return status;
}
