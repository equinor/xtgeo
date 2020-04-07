/*
 ***************************************************************************************
 *
 * NAME:
 *    grd3d_corners.c
 *
 * DESCRIPTION:
 *    Given a cell coordinate I J K, find all corner coordinates as an
 *    array with 24 values
 *
 *      Top  --> i-dir     Base cell
 *
 *  6,7,8   9,10,11  18,19,20   21,22,23      0 = X, 1 = Y, 2 = Z, etc
 *    |-------|          |-------|
 *    |       |          |       |
 *    |       |          |       |
 *    |-------|          |-------|
 *  0,1,2   3,4,5    12,13,14,  15,16,17
 *
 *
 * ARGUMENTS:
 *    i, j, k        i     Cell number
 *    nx,ny,nz       i     Grid dimensions
 *    coordsv        i     Grid Z coord for input
 *    zcornsv        i     Grid Z corners for input
 *    corners        o     Array, 24 length
 *
 * RETURNS:
 *    Corners
 *
 * TODO/ISSUES/BUGS:
 *    None known
 *
 * LICENCE:
 *    cf. XTGeo License
 ***************************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"
#include "logger.h"
#include <math.h>

void
grd3d_corners(int i,
              int j,
              int k,
              int nx,
              int ny,
              int nz,
              double *coordsv,
              long ncoordin,
              double *zcornsv,
              long nzcornin,
              double corners[])

{
    double xtop[5], ytop[5], ztop[5];
    double xbot[5], ybot[5], zbot[5];

    /* each cell is defined by 4 pillars */

    int ic;
    for (ic = 1; ic <= 4; ic++) {
        int jm = 0;
        int im = 0;
        if (ic == 1 || ic == 2)
            jm = 1;
        if (ic == 1 || ic == 3)
            im = 1;

        xtop[ic] = coordsv[6 * ((j - jm) * (nx + 1) + i - im) + 0];
        ytop[ic] = coordsv[6 * ((j - jm) * (nx + 1) + i - im) + 1];
        ztop[ic] = coordsv[6 * ((j - jm) * (nx + 1) + i - im) + 2];
        xbot[ic] = coordsv[6 * ((j - jm) * (nx + 1) + i - im) + 3];
        ybot[ic] = coordsv[6 * ((j - jm) * (nx + 1) + i - im) + 4];
        zbot[ic] = coordsv[6 * ((j - jm) * (nx + 1) + i - im) + 5];
    }

    /* cell and cell below*/
    long ibt = x_ijk2ib(i, j, k, nx, ny, nz + 1, 0);
    long ibb = x_ijk2ib(i, j, k + 1, nx, ny, nz + 1, 0);

    corners[2] = zcornsv[4 * ibt + 1 * 1 - 1];
    corners[5] = zcornsv[4 * ibt + 1 * 2 - 1];
    corners[8] = zcornsv[4 * ibt + 1 * 3 - 1];
    corners[11] = zcornsv[4 * ibt + 1 * 4 - 1];

    corners[14] = zcornsv[4 * ibb + 1 * 1 - 1];
    corners[17] = zcornsv[4 * ibb + 1 * 2 - 1];
    corners[20] = zcornsv[4 * ibb + 1 * 3 - 1];
    corners[23] = zcornsv[4 * ibb + 1 * 4 - 1];

    for (ic = 1; ic <= 8; ic++) {
        int cl = ic;
        if (ic == 5)
            cl = 1;
        if (ic == 6)
            cl = 2;
        if (ic == 7)
            cl = 3;
        if (ic == 8)
            cl = 4;

        if (fabs(zbot[cl] - ztop[cl]) > 0.01) {
            corners[3 * (ic - 1) + 0] =
              xtop[cl] - (corners[3 * (ic - 1) + 2] - ztop[cl]) *
                           (xtop[cl] - xbot[cl]) / (zbot[cl] - ztop[cl]);
            corners[3 * (ic - 1) + 1] =
              ytop[cl] - (corners[3 * (ic - 1) + 2] - ztop[cl]) *
                           (ytop[cl] - ybot[cl]) / (zbot[cl] - ztop[cl]);
        } else {
            corners[3 * (ic - 1) + 0] = xtop[cl];
            corners[3 * (ic - 1) + 1] = ytop[cl];
        }
    }
}
