/*
 ***************************************************************************************
 *
 * NAME:
 *    grd3d_point_val_crange
 *
 * DESCRIPTION:
 *    From a given point in XYZ, find the cell value. This routine assumes that
 *    I, J range is limited to speed up. It also uses a proposed I, J as start point.
 *    This routine is applied when sampling a fence from a 3D grid parameter
 *
 * ARGUMENTS:
 *
 *     x,y,z           i      input points
 *     nx..nz          i      grid dimensions
 *     coordsv         i      grid coords
 *     zcornsv         i      grid zcorn
 *     actnumsv        i      grid active cell indicator
 *     p_prop_v        i      property to work with
 *     value           o      value to return (UNDEF if not found)
 *     imin,.. kmax    i      cell index ranges in I J K
 *     ibs            i/o     Start point cell index, updated to actual cell index
 *     option          i      If negative, do not compute value (p_prop_v can be dummy)

 * RETURNS:
 *     int: < 0, then outside grid
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ***************************************************************************************
 */
#include <stdlib.h>
#include <xtgeo/xtgeo.h>
#include "common.h"

/*
 ***************************************************************************************
 * Private function
 ***************************************************************************************
 */

long
_find_ib(double x,
         double y,
         double z,
         int imin,
         int imax,
         int jmin,
         int jmax,
         int kmin,
         int kmax,
         int nx,
         int ny,
         int nz,
         double *coordsv,
         double *zcornsv)
{
    /* cell not found -9; otherwise >= 0 but active may be 0 if inactive cell*/

    int i, j, k, inside;
    long ib = -1;

    for (k = kmin; k <= kmax; k++) {
        for (j = jmin; j <= jmax; j++) {
            for (i = imin; i <= imax; i++) {

                double corners[24];
                ib = x_ijk2ib(i, j, k, nx, ny, nz, 0);
                if (ib < 0) {
                    throw_exception("Loop resulted in index outside "
                                    "boundary in grd3d_point_val_crange");
                    return -8;
                }
                /* get the corner for the cell */
                grd3d_corners(i, j, k, nx, ny, nz, coordsv, 0, zcornsv, 0, corners);
                inside = x_chk_point_in_cell(x, y, z, corners, 1);

                if (inside > 0) {
                    return ib;
                }
            }
        }
    }
    return -9;
}

/*
 ***************************************************************************************
 * Public function
 ***************************************************************************************
 */

int
grd3d_point_val_crange(double x,
                       double y,
                       double z,
                       int nx,
                       int ny,
                       int nz,
                       double *coordsv,
                       double *zcornsv,
                       int *actnumsv,
                       double *p_val_v,
                       double *value,
                       int imin,
                       int imax,
                       int jmin,
                       int jmax,
                       int kmin,
                       int kmax,
                       long *ibs,
                       int option)

{
    /* locals */
    long ibstart = 0, ib;
    int istart, jstart, kstart;
    int i1, i2, j1, j2, k1, k2;

    ibstart = *ibs;

    if (ibstart < 0) {
        ibstart = x_ijk2ib(imin, jmin, kmin, nx, ny, nz, 0);
        if (ibstart < 0) {
            throw_exception("Start index outside boundary in grd3d_point_val_crange");
            return -1;
        }
    }

    x_ib2ijk(ibstart, &istart, &jstart, &kstart, nx, ny, nz, 0);

    *value = UNDEF;

    /* first try, loop over a limited cells in range based on start*/
    k1 = kstart - 1;
    if (k1 < kmin)
        k1 = kmin;
    k2 = kstart + 1;
    if (k2 > kmax)
        k2 = kmax;
    j1 = jstart - 1;
    if (j1 < jmin)
        j1 = jmin;
    j2 = jstart + 1;
    if (j2 > jmax)
        j2 = jmax;
    i1 = istart - 1;
    if (i1 < imin)
        i1 = imin;
    i2 = istart + 1;
    if (i2 > imax)
        i2 = imax;

    ib = _find_ib(x, y, z, i1, i2, j1, j2, k1, k2, nx, ny, nz, coordsv, zcornsv);

    /* second try; fall back loop over fuller range */
    if (ib == -9) {
        ib = _find_ib(x, y, z, imin, imax, jmin, jmax, kmin, kmax, nx, ny, nz, coordsv,
                      zcornsv);
    }

    if (ib > 0) {
        *ibs = ib;
        if (option < 0)
            return EXIT_SUCCESS;

        if (actnumsv[ib] == 1) {
            *value = p_val_v[ib];
        }
        return EXIT_SUCCESS;
    }

    return -1;
}
