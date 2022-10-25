/*
 ***************************************************************************************
 *
 * NAME:
 *    grdcp3d_corners.c
 *
 * DESCRIPTION:
 *    Given a c coordinate I J K, find all corner coordinates as an
 *    array with 24 values. For xtgformat=2 layout
 *
 *      Top  --> i-dir     Base c
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
 *    i, j, k          i     c number, base 0
 *    ncol,nrow,nlay   i     Grid dimensions nx ny nz
 *    coordsv          i     Grid Z coord for input
 *    zcornsv          i     Grid Z corners for input
 *    corners          o     Array, 24 length allocated at client.
 *
 * RETURNS:
 *    corners, _xtgformat=2
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

void
grdcp3d_corners(long ic,
                long jc,
                long kc,
                long ncol,
                long nrow,
                long nlay,
                double *coordsv,
                long ncoordin,
                float *zcornsv,
                long nzcornin,
                double corners[])

{
    double coor[4][6];
    double zc[8];

    // number of nodes
    long nnrow = nrow + 1;
    long nnlay = nlay + 1;
    /* each cell is defined by 4 pillars */

    long nn = 0;
    long i, j, k;
    for (j = 0; j < 2; j++) {
        for (i = 0; i < 2; i++) {
            for (k = 0; k < 6; k++) {
                coor[nn][k] = coordsv[(ic + i) * nnrow * 6 + (jc + j) * 6 + k];
            }
            nn++;
        }
    }

    zc[0] = zcornsv[((ic + 0) * nnrow * nnlay + (jc + 0) * nnlay + (kc + 0)) * 4 + 3];
    zc[1] = zcornsv[((ic + 1) * nnrow * nnlay + (jc + 0) * nnlay + (kc + 0)) * 4 + 2];
    zc[2] = zcornsv[((ic + 0) * nnrow * nnlay + (jc + 1) * nnlay + (kc + 0)) * 4 + 1];
    zc[3] = zcornsv[((ic + 1) * nnrow * nnlay + (jc + 1) * nnlay + (kc + 0)) * 4 + 0];

    zc[4] = zcornsv[((ic + 0) * nnrow * nnlay + (jc + 0) * nnlay + (kc + 1)) * 4 + 3];
    zc[5] = zcornsv[((ic + 1) * nnrow * nnlay + (jc + 0) * nnlay + (kc + 1)) * 4 + 2];
    zc[6] = zcornsv[((ic + 0) * nnrow * nnlay + (jc + 1) * nnlay + (kc + 1)) * 4 + 1];
    zc[7] = zcornsv[((ic + 1) * nnrow * nnlay + (jc + 1) * nnlay + (kc + 1)) * 4 + 0];

    double p0[3], p1[3];

    long l, c;
    long ncn = 0;
    long cz = 0;
    for (l = 0; l < 2; l++) {
        for (c = 0; c < 4; c++) {
            p0[0] = coor[c][0];
            p0[1] = coor[c][1];
            p0[2] = coor[c][2];
            p1[0] = coor[c][3];
            p1[1] = coor[c][4];
            p1[2] = coor[c][5];

            double x, y;
            if (x_linint3d(p0, p1, zc[cz], &x, &y) == 0) {
                corners[ncn++] = x;
                corners[ncn++] = y;
                corners[ncn++] = zc[cz];
            } else {
                // coord lines are collapsed
                corners[ncn++] = p0[0];
                corners[ncn++] = p0[1];
                corners[ncn++] = zc[cz];
            }
            cz++;
        }
    }
}
