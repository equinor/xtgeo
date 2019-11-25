/*
 ******************************************************************************
 *
 * Finding extreme Z for a cell i j k
 *
 ******************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"

/*
 ******************************************************************************
 *
 * NAME:
 *    grd3d_zminmax.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Find the upper and lower Z coordinate of a given cell
 *
 * ARGUMENTS:
 *    i, j, k        i     Cell number (1 based)
 *    nx, ny, nz     i     Grid dimensions
 *    p_zcorn_v      i     Z coordinates
 *    option         i     Option: 0 return minimum, 1 return maximum
 *    debug          i     Debug level
 *
 * RETURNS:
 *    ZMIN or ZMAX for option 0 or 1
 *
 * TODO/ISSUES/BUGS:
 *    Stricter checks
 *
 * LICENCE:
 *    cf. XTGeo license
 ******************************************************************************
 */
double grd3d_zminmax(
                     int i,
                     int j,
                     int k,
                     int nx,
                     int ny,
                     int nz,
                     double *p_zcorn_v,
                     int option,
                     int debug
                     )

{
    int ic;
    long ibb, ibt;
    double zmin, zmax, zval;
    char sbn[24] = "grd3d_zminmax";

    xtgverbose(debug);

    if (debug > 2) xtg_speak(sbn, 3, "Enter %s", sbn);

    /* cell and cell below*/
    ibt = x_ijk2ib(i,j,k,nx,ny,nz+1,0);
    ibb = x_ijk2ib(i,j,k+1,nx,ny,nz+1,0);

    if (ibb < 0 || ibt < 0) {
        xtg_error(sbn, "Error in routine %s", sbn);
    }

    if (option == 0) {
        zmin = p_zcorn_v[4*ibt + 1*1 - 1];
        for (ic = 2; ic < 5; ic++) {
            zval = p_zcorn_v[4 * ibt + 1 * ic - 1];
            if (zval < zmin) zmin = zval;
        }
        return zmin;
    }
    else if (option == 1) {
        zmax = p_zcorn_v[4*ibb + 1*1 - 1];
        for (ic = 2; ic < 5; ic++) {
            zval = p_zcorn_v[4 * ibb + 1 * ic - 1];
            if (zval > zmax) zmax = zval;
        }
        return zmax;
    }
    else{
        return UNDEF;
    }
}
