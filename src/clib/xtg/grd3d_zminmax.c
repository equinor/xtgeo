/*
 ***************************************************************************************
 *
 * NAME:
 *    grd3d_zminmax.c
 *
 * DESCRIPTION:
 *    Find the upper and lower Z coordinate of a given cell
 *
 * ARGUMENTS:
 *    i, j, k        i     Cell number (1 based)
 *    nx, ny, nz     i     Grid dimensions
 *    zcornsv        i     Z coordinates
 *    option         i     Option: 0 return minimum, 1 return maximum
 *
 * RETURNS:
 *    ZMIN or ZMAX for option 0 or 1
 *
 * TODO/ISSUES/BUGS:
 *    Stricter checks
 *
 * LICENCE:
 *    cf. XTGeo license
 ***************************************************************************************
 */
#include "libxtg.h"
#include "libxtg_.h"
#include "logger.h"

double
grd3d_zminmax(int i, int j, int k, int nx, int ny, int nz, double *zcornsv, int option)

{
    int ic;
    long ibb, ibt;
    double zmin, zmax, zval;

    /* cell and cell below*/
    ibt = x_ijk2ib(i, j, k, nx, ny, nz + 1, 0);
    ibb = x_ijk2ib(i, j, k + 1, nx, ny, nz + 1, 0);
    if (ibt < 0 || ibb < 0) {
        throw_exception("Index outside boundary in grd3d_zminmax");
        return -1;
    }

    if (option == 0) {
        zmin = zcornsv[4 * ibt + 1 * 1 - 1];
        for (ic = 2; ic < 5; ic++) {
            zval = zcornsv[4 * ibt + 1 * ic - 1];
            if (zval < zmin)
                zmin = zval;
        }
        return zmin;
    } else if (option == 1) {
        zmax = zcornsv[4 * ibb + 1 * 1 - 1];
        for (ic = 2; ic < 5; ic++) {
            zval = zcornsv[4 * ibb + 1 * ic - 1];
            if (zval > zmax)
                zmax = zval;
        }
        return zmax;
    } else {
        return UNDEF;
    }
}
