#include "libxtg.h"
#include "libxtg_.h"
#include "logger.h"
#include <math.h>

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

    double ap = pow(a, 2);
    double bp = pow(b, 2);
    double cp = pow(c, 2);
    double dp = pow(d, 2);
    double ep = pow(e, 2);
    double fp = pow(f, 2);

    double vol = 4 * (cp * ap * bp) - cp * pow((ap + bp - dp), 2) -
                 ap * pow((bp + cp - fp), 2) - bp * pow((cp + ap - ep), 2) +
                 (ap + bp - dp) * (bp + cp - fp) * (cp + ap - ep);

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
 *
 * RETURNS:
 *    1 if inside, 0 else
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ***************************************************************************************
 */

int
x_point_in_tetrahedron(double x0, double y0, double z0, double *pv, long ndim)
{

    double truevol = x_tetrahedron_volume(pv, 12);

    int i, nv;
    double newpv[12];
    double sumvol;

    sumvol = 0.0;
    for (nv = 0; nv < 4; nv++) {
        /* code */
        for (i = 0; i < ndim; i++) {
            newpv[i] = pv[i];
        }
        newpv[0 + 3 * nv] = x0;
        newpv[1 + 3 * nv] = y0;
        newpv[2 + 3 * nv] = z0;
        double vol = x_tetrahedron_volume(newpv, ndim);
        sumvol += vol;
    }

    double relerror = truevol * 0.0001;
    double diff = sumvol - truevol;
    if (diff < -1 * relerror) {
        logger_critical(LI, FI, FU, "Something is wrong in %s!", FU);
        printf("Something is rotten sumvol vs total %lf %lf\n", sumvol, truevol);
        return -1;
    } else if (diff > relerror) {
        return 0;
    } else {
        return 1;
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
    double **crn = x_allocate_2d_double(8, 3);
    int **cset = x_allocate_2d_int(4, 5);

    int i, j;
    int ic = 0;
    for (i = 0; i < 8; i++) {
        for (j = 0; j < 3; j++) {
            crn[i][j] = corners[ic++];
        }
    }

    // the hexehedron consists of 5 tetrahedrons

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

    cset[0][4] = 5;
    cset[1][4] = 6;
    cset[2][4] = 0;
    cset[3][4] = 3;

    int icset;
    double thd[12];

    int status = 0;
    for (icset = 0; icset < 5; icset++) {
        ic = 0;
        for (i = 0; i < 4; i++) {
            thd[ic + 0] = crn[cset[i][icset]][0];
            thd[ic + 1] = crn[cset[i][icset]][1];
            thd[ic + 2] = crn[cset[i][icset]][2];
            ic += 3;
        }
        if (x_point_in_tetrahedron(x0, y0, z0, thd, 12) == 1) {
            status += 50;
            break;
        }
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

    for (icset = 0; icset < 5; icset++) {
        ic = 0;
        for (i = 0; i < 4; i++) {
            thd[ic + 0] = crn[cset[i][icset]][0];
            thd[ic + 1] = crn[cset[i][icset]][1];
            thd[ic + 2] = crn[cset[i][icset]][2];
            ic += 3;
        }
        if (x_point_in_tetrahedron(x0, y0, z0, thd, 12) == 1) {
            status += 50;
            break;
        }
    }

    x_free_2d_double(crn);
    x_free_2d_int(cset);

    return status;
}
