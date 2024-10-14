/*
 ***************************************************************************************
 *
 * NAME:
 *    grd3d_point_in_cell.c
 *
 * DESCRIPTION:
 *    Find which cell (ib) that contains the current point. An input ib gives
 *    a much faster search if the next point is close to the first one.
 *
 *    Note there are several variant of this in the library, as the need for speed
 *    is most chellenging
 *
 * ARGUMENTS:
 *    ibstart          which IB to start search from
 *    kzonly           A number in [1..nz] if only looking within a
 *                      cell layer, 0 otherwise
 *    x,y,z            input points. If z is -999 it means a 2D search only
 *    nx..nz           grid dimensions
 *    coordsv          grid coords
 *    zcornsv          grid zcorn
 *    actnumsv         grid active cell indicator
 *    p_prop_v         property to work with
 *    value            value to set
 *    ronly            replace-only-this value
 *    i1, i2...k2      cell index range in I J K
 *    option           0 for looking in cell 3D, 1 for looking in 2D bird view
 *
 * RETURNS:
 *    A value ranging from 0 to nx*ny*nz-1 of found.
 *
 * TODO/ISSUES/BUGS:
 *    * consistency if ibstart > 0 but not in correct kzonly layer?
 *    * how to use ACTNUM (not used so far)
 *
 * LICENCE:
 *    CF XTGeo's LICENSE
 ***************************************************************************************
 */
#include <xtgeo/xtgeo.h>
#include "common.h"
#include "logger.h"

long
grd3d_point_in_cell(long ibstart,
                    int kzonly,
                    double x,
                    double y,
                    double z,
                    int nx,
                    int ny,
                    int nz,
                    double *p_coor_v,
                    double *zcornsv,
                    int *actnumsv,
                    int maxrad,
                    int sflag,
                    int *nradsearch,
                    int option)

{
    /* locals */
    // long ib;
    // int i, j, k, inside, irad;
    // int i1, i2, j1, j2, k1, k2;

    logger_info(LI, FI, FU, "Finding if point in cell: %s, ibstart %ld", FU, ibstart);

    if (ibstart < 0)
        ibstart = 0;

    if (kzonly > 0 && ibstart == 0) {
        ibstart = x_ijk2ib(1, 1, kzonly, nx, ny, nz, 0);
        if (ibstart < 0) {
            throw_exception("Outside cell in grd3d_point_in_cell");
            return -1;
        }
    }

    int istart, jstart, kstart;
    x_ib2ijk(ibstart, &istart, &jstart, &kstart, nx, ny, nz, 0);

    /*
     * Will search in a growing radius around a start point
     * in order to optimize speed
     */

    int i1 = istart;
    int j1 = jstart;
    int k1 = kstart;

    int i2 = istart;
    int j2 = jstart;
    int k2 = kstart;

    int irad;
    for (irad = 0; irad <= (maxrad + 1); irad++) {

        if (irad > 0) {
            i1 -= 1;
            i2 += 1;
            j1 -= 1;
            j2 += 1;
            k1 -= 1;
            k2 += 1;
        }

        if (sflag > 0 && irad > maxrad) {
            i1 = 1;
            i2 = nx;
            j1 = 1;
            j2 = ny;
            k1 = 1;
            k2 = nz;
        }

        *nradsearch = irad;

        if (i1 < 1)
            i1 = 1;
        if (j1 < 1)
            j1 = 1;
        if (k1 < 1)
            k1 = 1;
        if (i2 > nx)
            i2 = nx;
        if (j2 > ny)
            j2 = ny;
        if (k2 > nz)
            k2 = nz;

        if (kzonly > 0) {
            k1 = kzonly;
            k2 = kzonly;
        }

        int i, j, k;
        for (k = k1; k <= k2; k++) {
            for (j = j1; j <= j2; j++) {
                for (i = i1; i <= i2; i++) {

                    long ib = x_ijk2ib(i, j, k, nx, ny, nz, 0);
                    if (ib < 0) {
                        throw_exception("Loop resulted in index outside "
                                        "boundary in grd3d_point_in_cell");
                        return -1;
                    }
                    double corners[24];
                    double polx[5], poly[5];

                    /* get the corner for the cell */
                    grd3d_corners(i, j, k, nx, ny, nz, p_coor_v, 0, zcornsv, 0,
                                  corners);

                    int inside = 0;
                    if (option == 0) {
                        /* 3D cell */
                        inside = x_chk_point_in_cell(x, y, z, corners, 1);

                    } else {
                        /*2D view ...  make a closed polygon in XY
                          from the corners */
                        polx[0] = 0.5 * (corners[0] + corners[12]);
                        poly[0] = 0.5 * (corners[1] + corners[13]);
                        polx[1] = 0.5 * (corners[3] + corners[15]);
                        poly[1] = 0.5 * (corners[4] + corners[16]);
                        polx[2] = 0.5 * (corners[9] + corners[21]);
                        poly[2] = 0.5 * (corners[10] + corners[22]);
                        polx[3] = 0.5 * (corners[6] + corners[18]);
                        poly[3] = 0.5 * (corners[7] + corners[19]);
                        polx[4] = polx[0];
                        poly[4] = poly[0];

                        inside =
                          pol_chk_point_inside((double)x, (double)y, polx, poly, 5);
                    }

                    if (inside > 0) {
                        return ib;
                    }
                }
            }
        }

        if (i1 == 1 && i2 == nx && j1 == 1 && j2 == ny && k1 == 1 && k2 == nz)
            break;
    }

    return -1; /* if nothing found */
}
