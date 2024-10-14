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
#include <math.h>
#include <stdlib.h>
#include <xtgeo/xtgeo.h>
#include "logger.h"

static struct
{
    long icol;
    long jrow;
    long klay;
    long ncol;
    long nrow;
    long nlay;
    long icount;
    long ncount;
    double *coordsv;
    long ncoord;
    float *zcornsv;
    long nzcorn;
    int *actnum;
    double *corners;
    float *fresults;
} data;

static void
_cellangles()
{
    /* update the result vector with angle data */
    double amin, amax, aminp, amaxp, amins, amaxs;

    int ier1 = x_minmax_cellangles_topbase(data.corners, 24, &amin, &amax, 0, 1);
    int ier2 = x_minmax_cellangles_topbase(data.corners, 24, &aminp, &amaxp, 1, 1);
    int ier3 = x_minmax_cellangles_sides(data.corners, 24, &amins, &amaxs, 1);
    if (ier1 != 0) {
        amin = UNDEF;
        amax = UNDEF;
    }
    if (ier2 != 0) {
        aminp = UNDEF;
        amaxp = UNDEF;
    }
    if (ier3 != 0) {
        amins = UNDEF;
        amaxs = UNDEF;
    }

    // if (data.actnum[data.icount] == 0) {
    //     amin = -999;
    //     amax = -999;
    // }

    data.ncount = data.ncol * data.nrow * data.nlay;

    data.fresults[data.ncount * 0 + data.icount] = (float)amin;
    data.fresults[data.ncount * 1 + data.icount] = (float)amax;
    data.fresults[data.ncount * 2 + data.icount] = (float)aminp;
    data.fresults[data.ncount * 3 + data.icount] = (float)amaxp;
    data.fresults[data.ncount * 4 + data.icount] = (float)amins;
    data.fresults[data.ncount * 5 + data.icount] = (float)amaxs;
}

static void
_collapsed()
{
    // Cells are collapsed vertically in one or more corners

    int ic;
    float iscollapsed = 0.0;
    for (ic = 2; ic < 12; ic += 3) {
        if (fabs(data.corners[ic + 12] - data.corners[ic]) < FLOATEPS) {
            iscollapsed = 1.0;
            break;
        }
    }

    data.fresults[data.ncount * 6 + data.icount] = iscollapsed;
}

static void
_faulted()
{
    // Detect faulted splitnodes

    int ic, jc, kc;
    float isfaulted = 0.0;
    long nnrow = data.nrow + 1;
    long nnlay = data.nlay + 1;

    for (ic = 0; ic < 2; ic++) {
        for (jc = 0; jc < 2; jc++) {
            for (kc = 0; kc < 2; kc++) {
                long nc = 4 * ((data.icol + ic) * nnrow * nnlay +
                               (data.jrow + jc) * nnlay + (data.klay + kc));
                double sw = data.zcornsv[nc + 0];
                double se = data.zcornsv[nc + 1];
                double nw = data.zcornsv[nc + 2];
                double ne = data.zcornsv[nc + 3];

                double diff[4];
                diff[0] = fabs(sw - se);
                diff[1] = fabs(sw - nw);
                diff[2] = fabs(sw - ne);
                diff[3] = fabs(se - nw);

                int i;
                for (i = 0; i < 4; i++) {
                    if (diff[i] > FLOATEPS) {
                        isfaulted = 1.0;
                        break;
                    }
                }
            }
        }
    }
    data.fresults[data.ncount * 7 + data.icount] = isfaulted;
}

static void
_negativethickness()
{
    // Detect negative thickness

    int i;
    float isnegative = 0.0;
    for (i = 2; i < 12; i += 3) {
        double zdiff = data.corners[i + 12] - data.corners[i];
        if (zdiff < 0.0)
            isnegative = 1.0;
    }

    data.fresults[data.ncount * 8 + data.icount] = isnegative;
}

static void
_concave_projected()
{
    // Detect if a cell is concave seen from above. A cell is concave if one corner
    // is within the triangle formed by the other corners at top and/or base. The
    // result is number; if 0 then cell is convex, if 1 then first corner is concave,
    // ... up to 4. Only X, Y coordinates are checked

    double xp[4][2];
    double yp[4][2];

    int n;
    for (n = 0; n < 4; n++) {
        xp[n][0] = data.corners[n * 3 + 0];
        yp[n][0] = data.corners[n * 3 + 1];
        xp[n][1] = data.corners[n * 3 + 0 + 12];
        yp[n][1] = data.corners[n * 3 + 1 + 12];
    }

    double xtri[4], ytri[4];

    int nchk, ntop;
    float collapse = 0.0;

    for (ntop = 0; ntop < 2; ntop++) {
        for (nchk = 0; nchk < 4; nchk++) {

            int inside;

            double xuse = xp[nchk][ntop];
            double yuse = yp[nchk][ntop];
            int nc = 0;
            for (n = 0; n < 4; n++) {
                if (n != nchk) {
                    xtri[nc] = xp[n][ntop];
                    ytri[nc] = yp[n][ntop];
                    nc++;
                }
            }
            xtri[3] = xtri[0];
            ytri[3] = ytri[0];
            inside = pol_chk_point_inside(xuse, yuse, xtri, ytri, 4);
            if (inside >= 1) {
                collapse = nchk + 1.0;
                goto RESULT;
            }
        }
    }

RESULT:
    data.fresults[data.ncount * 9 + data.icount] = collapse;
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
    double *corners = calloc(24, sizeof(double));

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

    data.corners = corners;
    data.ncount = data.ncol * data.nrow * data.nlay;

    long i, j, k;
    for (i = 0; i < ncol; i++) {
        for (j = 0; j < nrow; j++) {
            for (k = 0; k < nlay; k++) {

                long ic = i * nrow * nlay + j * nlay + k;

                data.icol = i;
                data.jrow = j;
                data.klay = k;
                data.icount = ic;
                grdcp3d_corners(data.icol, data.jrow, data.klay, data.ncol, data.nrow,
                                data.nlay, data.coordsv, data.ncoord, data.zcornsv,
                                data.nzcorn, data.corners);
                _cellangles();
                _collapsed();
                _faulted();
                _negativethickness();
                _concave_projected();
            }
        }
        logger_info(LI, FI, FU, "Grid quality measures... done");
    }
    free(corners);
}
