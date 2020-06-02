/*
 ***************************************************************************************
 *
 * NAME:
 *    grd3d_points_ijk_cells.c
 *
 *
 * DESCRIPTION:
 *    Given X Y Z vectors, return the corresponding I J K vectors for the cell indices
 *    Certain tricks here are made in order to get it work fast:
 *    > Define maps with IJK indixes to search for I J ranges
 *    > A onelayer version of the grid
 *
 * ARGUMENTS:
 *    xvec, yvec, zvec    i     Arrays coords XYZ
 *    n*vec               i     length of input vectors (spesified for swig/numpy)
 *    mcol, mrow          i     Number of rows/cols for maps
 *    xori..rotation      i     Map settings
 *    maptopi..mapbasj    i     Map arrays for I J top/base
 *    nx ny nz            i     Grid dimensions
 *    zcornsv             i     Grid ZCORN
 *    coordsv             i     Grid COORD
 *    p_acnum_v           i     Grid ACTNUM
 *    p_val_v             i     3D Grid values
 *    p_zcornone_v        i     Grid ZCORN for onelayer grid
 *    p_acnumone_v        i     Grid ACTNUM for onelayer grid
 *    actnumoption        i     if 1, then only report if cell is active
 *    flip                i     1, or -1 if right handed with K down
 *    ivec, jvec, kvec    o     IJK arrays
 *    n*vec               i     array lengths (for swig/numpies)
 *    tolerance           i     Tolerance of matching grid cells (0..1)
 *
 * RETURNS:
 *    Update IJK pointers, array length, -1 if fail
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ***************************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"
#include "logger.h"

#include <math.h>

/*
****************************************************************************************
* private functions
****************************************************************************************
*/

static void
_get_ij_range2(int *i1,
               int *i2,
               int *j1,
               int *j2,
               double xc,
               double yc,
               int mcol,
               int mrow,
               double xori,
               double yori,
               double xinc,
               double yinc,
               int yflip,
               double rotation,
               double *maptopi,
               double *maptopj,
               double *mapbasi,
               double *mapbasj,
               int nx,
               int ny)
{
    long nmap;
    int itop, jtop, ibas, jbas, ii1, ii2, jj1, jj2;

    nmap = mcol * mrow;

    /* get map value for I J from x y */
    itop = surf_get_z_from_xy(xc, yc, mcol, mrow, xori, yori, xinc, yinc, yflip,
                              rotation, maptopi, nmap);
    jtop = surf_get_z_from_xy(xc, yc, mcol, mrow, xori, yori, xinc, yinc, yflip,
                              rotation, maptopj, nmap);
    ibas = surf_get_z_from_xy(xc, yc, mcol, mrow, xori, yori, xinc, yinc, yflip,
                              rotation, mapbasi, nmap);
    jbas = surf_get_z_from_xy(xc, yc, mcol, mrow, xori, yori, xinc, yinc, yflip,
                              rotation, mapbasj, nmap);

    if (itop <= ibas) {
        ii1 = itop;
        ii2 = ibas;
    } else {
        ii1 = ibas;
        ii2 = itop;
    }

    /* extend with one to avoid edge effects missing values */
    if (ii1 > 1)
        ii1--;
    if (ii2 < mcol)
        ii2++;

    if (jtop <= jbas) {
        jj1 = jtop;
        jj2 = jbas;
    } else {
        jj1 = jbas;
        jj2 = jtop;
    }

    /* extend with one to avoid edge effects missing values */
    if (jj1 > 1)
        jj1--;
    if (jj2 < mrow)
        jj2++;

    /*  if numbers are unphysical for some reason, revert to grid limits */
    if (ii1 < 1 || ii1 >= nx)
        ii1 = 1;
    if (ii2 <= 1 || ii2 > nx)
        ii2 = nx;
    if (jj1 < 1 || jj1 >= ny)
        jj1 = 1;
    if (jj2 <= 1 || jj2 > ny)
        jj2 = ny;

    if (ii2 <= ii1 || (ii2 - ii1) >= nx || jj2 <= jj1 || (jj2 - jj1) >= ny) {
        ii1 = 1;
        ii2 = nx;
        jj1 = 1;
        jj2 = ny;
    }

    *i1 = ii1;
    *i2 = ii2;
    *j1 = jj1;
    *j2 = jj2;
}

static long
_grd3d_point_in_cell(int ic,
                     int jc,
                     int kc,
                     double xc,
                     double yc,
                     double zc,
                     int nx,
                     int ny,
                     int nz,
                     double *p_coor_v,
                     double *zcornsv,
                     int *score,
                     int flip,
                     int tolopt,
                     int dbg)

{
    /*
     * This is a special version of the finding a point in cell, optimised
     * for this work flow.
     *
     * ic, jc, kc    Proposed cell I J (estimated by previous steps in work flow)
     * xc, yc, zc    Point to evaluate if inside cell
     * nx, ny, nz    Dimensions
     * p_coor_v      Coordinates (grid)
     * zcornsv       ZCORN (grid)
     * score         This is a number telling how good the match is, going from -1 to 24
     */

    long ib;

    ib = x_ijk2ib(ic, jc, kc, nx, ny, nz, 0);

    /* get the corner for the cell */
    double corners[24];
    grd3d_corners(ic, jc, kc, nx, ny, nz, p_coor_v, 0, zcornsv, 0, corners);

    *score = x_chk_point_in_hexahedron(xc, yc, zc, corners, flip);

    if (*score > tolopt) {
        return ib;
    }

    return -1; /* if nothing found */
}

static long
_point_val_ij(double xc,
              double yc,
              double zc,
              int nx,
              int ny,
              double *p_coor_v,
              double *p_zcornone_v,
              int i1,
              int i2,
              int j1,
              int j2,
              int flip,
              int tolopt,
              int dbg)
{
    /*
     * The purpose here is to search the one layer grid for IJ location of point XYZ
     * In case inside, IRES and JRES are assigned; otherwise, it will be UNDEF_INT
     * This routine should be fast since the I1 I2 J1 J2 range is estimated in the
     * previous step
     *
     * ires, jres       Pointers to I J to be updated
     * xc, yc, zc       Points to evaluate if inside
     * nx, ny, nz       Dimensions
     * p_coor_v         Coordinates COORD
     * zcornsv        Coordinates ZCORN
     * i1, i2, j1, j2   I J search range
     */

    int ii, jj;

    int score;

    long ib_alternatives[24];
    int ibn;
    for (ibn = 0; ibn < 24; ibn++)
        ib_alternatives[ibn] = -1;

    int nnn = 0;
    for (jj = j1; jj <= j2; jj++) {
        for (ii = i1; ii <= i2; ii++) {

            score = 0;
            long ibfound =
              _grd3d_point_in_cell(ii, jj, 1, xc, yc, zc, nx, ny, 1, p_coor_v,
                                   p_zcornone_v, &score, flip, tolopt, dbg);

            if (score > tolopt) {
                ib_alternatives[nnn++] = ibfound;
            }
        }
    }

    if (nnn > 0) {
        return ib_alternatives[nnn / 2];
    }

    return -1;
}

static long
_point_val_ijk(double xc,
               double yc,
               double zc,
               int nx,
               int ny,
               int nz,
               double *p_coor_v,
               double *zcornsv,
               int iin,
               int jin,
               int flip,
               int tolopt,
               int dbg)
{
    /*
     * The purpose here is to search the final layer grid for K location of point XYZ
     *
     * xc, yc, zc       Points to evaluate if inside
     * nx, ny, nz       Dimensions
     * p_coor_v         Coordinates COORD
     * zcornsv        Coordinates ZCORN
     * iin, jin         I J column
     */

    int score;

    long ib_alternatives[24];
    int ibn;
    for (ibn = 0; ibn < 24; ibn++)
        ib_alternatives[ibn] = -1;

    int k;
    int nnn = 0;
    for (k = 1; k <= nz; k++) {
        long ibfound =
          _grd3d_point_in_cell(iin, jin, k, xc, yc, zc, nx, ny, nz, p_coor_v, zcornsv,
                               &score, flip, tolopt, dbg);

        if (score > tolopt) {
            ib_alternatives[nnn++] = ibfound;
        }
    }

    if (nnn > 0) {
        return ib_alternatives[nnn / 2];
    }

    return -1;
}

/*
****************************************************************************************
* public function
****************************************************************************************
*/

int
grd3d_points_ijk_cells(double *xvec,
                       long nxvec,
                       double *yvec,
                       long nyvec,
                       double *zvec,
                       long nzvec,

                       int mcol,
                       int mrow,
                       double xori,
                       double yori,
                       double xinc,
                       double yinc,
                       double rotation,
                       int yflip,
                       double *maptopi,
                       double *maptopj,
                       double *mapbasi,
                       double *mapbasj,

                       int nx,
                       int ny,
                       int nz,

                       double *p_coor_v,
                       long ncoordin,
                       double *zcornsv,
                       long nzcornin,
                       int *actnumsv,
                       long nactin,

                       double *p_zcornone_v,
                       long nzcornonein,

                       int actnumoption,
                       int flip,

                       int *ivec,
                       long nivec,
                       int *jvec,
                       long njvec,
                       int *kvec,
                       long nkvec,
                       float tolerance)
{

    logger_info(LI, FI, FU, "Entering routine %s", FU);

    if (nxvec != nyvec || nyvec != nzvec)
        logger_critical(LI, FI, FU, "Input bug");
    if (nivec != njvec || nivec != nkvec)
        logger_critical(LI, FI, FU, "Input bug");

    long ib = 0;

    int tolopt = x_nint(24 - (24 * tolerance));  // inverse

    int ic;
    for (ic = 0; ic < nxvec; ic++) {
        double xc = xvec[ic];
        double yc = yvec[ic];
        double zc = zvec[ic];

        int dbg = 0;
        if (fabs(xc - 456620.790918) < 0.0000001)
            dbg = 1;

        /*
         * first get an approximate I and J range based on these maps
         * This is based on xc, yc postion in the regular map
         */

        int i1, i2, j1, j2;
        _get_ij_range2(&i1, &i2, &j1, &j2, xc, yc, mcol, mrow, xori, yori, xinc, yinc,
                       yflip, rotation, maptopi, maptopj, mapbasi, mapbasj, nx, ny);

        /*
         * next check the onelayer version of the grid first (speed up)
         * This should pin I J coordinate
         */

        long ibfound = _point_val_ij(xc, yc, zc, nx, ny, p_coor_v, p_zcornone_v, i1, i2,
                                     j1, j2, flip, tolopt, dbg);

        ivec[ic] = UNDEF_INT;
        jvec[ic] = UNDEF_INT;
        kvec[ic] = UNDEF_INT;

        if (ibfound >= 0) {
            if (dbg)
                logger_info(LI, FI, FU, "Use Return IB %d", ib);
            /*
             * means that the  X Y Z point is somewhere inside
             * so now it is time to find exact K location
             */
            int ires, jres, kres;
            x_ib2ijk(ibfound, &ires, &jres, &kres, nx, ny, 1, 0);

            long ibfound2 = _point_val_ijk(xc, yc, zc, nx, ny, nz, p_coor_v, zcornsv,
                                           ires, jres, flip, tolopt, dbg);
            if (ibfound2 >= 0) {
                x_ib2ijk(ibfound2, &ires, &jres, &kres, nx, ny, nz, 0);
                ivec[ic] = ires;
                jvec[ic] = jres;
                kvec[ic] = kres;

                if (actnumoption == 1 && actnumsv[ibfound2] == 0) {
                    /*  reset to undef in inactivecell */
                    ivec[ic] = UNDEF_INT;
                    jvec[ic] = UNDEF_INT;
                    kvec[ic] = UNDEF_INT;
                }
            }
        }
    }

    logger_info(LI, FI, FU, "Exit from routine %s", FU);

    return EXIT_SUCCESS;
}
