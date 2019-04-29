/*
 ******************************************************************************
 *
 * A collection of 3D geomtrical vectors, planes, etc
 *
 ******************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"

/*
 ******************************************************************************
 *
 * NAME:
 *    grd3d_ijwindow_from_poly.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Given a layer K in the 3D grid, find the IJ window based on a polygon
 *
 * ARGUMENTS:
 *    np             i     Numper of points in polygon vector
 *    p_xp_v p_yp_v  i     Vector for X and Y coords of polygon
 *    nx .. nz       i     Grid dimensions
 *    klayer         i     Grid layer to work with
 *    p_coord_v      i     Grid COORD lines
 *    p_zcorn_v      i     Grid ZCORN
 *    p_actnum_v     i     Grid ACTNUM prop
 *    i1 .. j2       o     Return of grid IJ window
 *    extra          i     Extra distance in units (e.g. metres)
 *                         (to make the window larger or less)
 *    option         i     Options flag for later usage
 *    debug          i     Debug level
 *
 * RETURNS:
 *    Function: 0: upon success. If problems:
 *              -1: No window found
 *    Result i1 i2 j1 j2 are updated
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ******************************************************************************
 */

int grd3d_ijwindow_from_poly(
                             int np,
                             double *p_xp_v,
                             double *p_yp_v,
                             int nx,
                             int ny,
                             int nz,
                             int klayer,
                             double *p_coord_v,
                             double *p_zcorn_v,
                             int *p_actnum_v,
                             int *i1,
                             int *i2,
                             int *j1,
                             int *j2,
                             double extra,
                             int option,
                             int debug
                             )
{
    char s[24]="grd3d_ijwi.._from_poly";
    double xg, yg, zg;
    double xori, yori, zori, xmin, xmax, ymin, ymax;
    double zmin, zmax, rotation, dx, dy, dz;
    int i, j, ii1, ii2, jj1, jj2, istat, iextra, jextra;

    ii1 = nx + 1;
    ii2 = 0;
    jj1 = ny + 1;
    jj2 = 0;

    xtgverbose(debug);

    /* need the avg geometrics for DX DY */
    istat = grd3d_geometrics(nx ,ny, nz, p_coord_v, p_zcorn_v, p_actnum_v,
                             &xori, &yori, &zori, &xmin, &xmax, &ymin, &ymax,
                             &zmin, &zmax, &rotation, &dx, &dy, &dz, 0, 0,
                             debug);

    /* convert extra to approx cell numbers */

    iextra = x_nint(extra / dx);
    jextra = x_nint(extra / dy);

    xtg_speak(s,2,"Layer is %d",klayer);
    for (j = 1; j <= ny; j ++) {
        for (i = 1; i <= nx; i ++) {
            grd3d_midpoint(i, j, klayer, nx, ny, nz, p_coord_v,
                           p_zcorn_v, &xg, &yg, &zg, debug);


            /* search if XG, YG is present in polygon */
            istat=0;

            istat = pol_chk_point_inside(xg, yg, p_xp_v, p_yp_v, np, debug);

            if (istat > 0) xtg_speak(s,2,"ISTAT %d for %d %d", istat, i, j);

            if (istat > 0) {
                if (i < ii1) ii1 = i;
                if (i > ii2) ii2 = i;
                if (j < jj1) jj1 = j;
                if (j > jj2) jj2 = j;
                xtg_speak(s,2,"Range %d %d   %d %d", ii1, ii2, jj1, jj2);
            }
        }
    }

    if (ii1 > ii2 || jj1 > jj2) return(-1);

    if (iextra != 0 || jextra != 0) {
        ii1 -= iextra;
        ii2 += iextra;
        jj1 -= jextra;
        jj2 += jextra;

        if (ii1 < 1) ii1 = 1;
        if (jj1 < 1) jj1 = 1;
        if (ii2 > nx) ii2 = nx;
        if (jj2 > ny) jj2 = ny;
    }

    *i1 = ii1;
    *i2 = ii2;
    *j1 = jj1;
    *j2 = jj2;

    xtg_speak(s,2,"Exit from %s with status 0", s);
    return 0;
}
