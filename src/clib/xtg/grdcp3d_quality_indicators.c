/*
 ***************************************************************************************
 *
 * NAME:
 *    grdcp3d_quality_indicators.c
 *
 * DESCRIPTION:
 *    Generate a set of grid quality measures for a cell, such as:
 *    - min/max angle at top and base, normal and projected
 *    - collapsed cell corners
 *    - concave cells
 *    - etc
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
#include <math.h>

static struct
{
    long icol;
    long jrow;
    long klay;
    long ncol;
    long nrow;
    long nlay;
    long icount;
    double *coordsv;
    long ncoord;
    float *zcornsv;
    long nzcorn;
    int *actnum;
    float *fresults;
} data;

static void
_cellangles()
{
    /* update the result vector with angle data */
    double amin, amax, aminp, amaxp;
    double corners[24];
    grdcp3d_corners(data.icol, data.jrow, data.klay, data.ncol, data.nrow, data.nlay,
                    data.coordsv, data.ncoord, data.zcornsv, data.nzcorn, corners);
    int ier1 = x_minmax_cellangles(corners, 24, &amin, &amax, 0, 1);
    int ier2 = x_minmax_cellangles(corners, 24, &aminp, &amaxp, 1, 1);
    if (ier1 != 0) {
        amin = UNDEF;
        amax = UNDEF;
    }
    if (ier2 != 0) {
        aminp = UNDEF;
        amaxp = UNDEF;
    }

    // if (data.actnum[data.icount] == 0) {
    //     amin = -999;
    //     amax = -999;
    // }

    long ncount = data.ncol * data.nrow * data.nlay;

    data.fresults[ncount * 0 + data.icount] = amin;
    data.fresults[ncount * 1 + data.icount] = amax;
    data.fresults[ncount * 2 + data.icount] = aminp;
    data.fresults[ncount * 3 + data.icount] = amaxp;
}

/*
 * -------------------------------------------------------------------------------------
 * public function
 * -------------------------------------------------------------------------------------
 */

void
grdcp3d_quality_indicators(long ncol,
                           long nrow,
                           long nlay,
                           double *coordsv,
                           long ncoordin,
                           float *zcornsv,
                           long nzcornin,
                           int *actnumsv,
                           long nact,
                           float *fresults,
                           long nfresults)

{
    /* each cell is defined by 4 pillars */

    logger_info(LI, FI, FU, "Grid quality measures...");

    // struct data;

    data.ncol = ncol;
    data.nrow = nrow;
    data.nlay = nlay;
    data.coordsv = coordsv;
    data.ncoord = ncoordin;
    data.zcornsv = zcornsv;
    data.nzcorn = nzcornin;
    data.actnum = actnumsv;
    data.fresults = fresults;

    long i, j, k;
    for (i = 0; i < ncol; i++) {
        for (j = 0; j < nrow; j++) {
            for (k = 0; k < nlay; k++) {

                long ic = i * nrow * nlay + j * nlay + k;

                data.icol = i;
                data.jrow = j;
                data.klay = k;
                data.icount = ic;
                _cellangles(data);
            }
        }
        logger_info(LI, FI, FU, "Grid quality measures... done");
    }

    // fresults = data.fresults;
}