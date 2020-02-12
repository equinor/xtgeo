/*
 ***************************************************************************************
 *
 * NAME:
 *    grd3d_points_ijk_cells.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
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
 *    p_zcorn_v           i     Grid ZCORN
 *    p_coord_v           i     Grid COORD
 *    p_acnum_v           i     Grid ACTNUM
 *    p_val_v             i     3D Grid values
 *    p_zcornone_v        i     Grid ZCORN for onelayer grid
 *    p_acnumone_v        i     Grid ACTNUM for onelayer grid
 *
 *    ivec, jvec, kvec    o     IJK arrays
 *    n*vec               i     array lengths (for swig/numpies)
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

#include "logger.h"
#include "libxtg.h"
#include "libxtg_.h"


/*
****************************************************************************************
* private functions
****************************************************************************************
*/

void _get_ij_range2(int *i1,  int *i2, int *j1, int *j2, double xc, double yc, int mcol,
                    int mrow, double xori, double yori, double xinc, double yinc,
                    int yflip, double rotation, double *maptopi, double *maptopj,
                    double *mapbasi, double *mapbasj)
{
    long nmap;
    int itop, jtop, ibas, jbas, ii1, ii2, jj1, jj2;

    nmap = mcol * mrow;

    /* get map value for I J from x y */
    itop = surf_get_z_from_xy(xc, yc, mcol, mrow, xori, yori, xinc, yinc,
                              yflip, rotation, maptopi, nmap);
    jtop = surf_get_z_from_xy(xc, yc, mcol, mrow, xori, yori, xinc, yinc,
                              yflip, rotation, maptopj, nmap);
    ibas = surf_get_z_from_xy(xc, yc, mcol, mrow, xori, yori, xinc, yinc,
                              yflip, rotation, mapbasi, nmap);
    jbas = surf_get_z_from_xy(xc, yc, mcol, mrow, xori, yori, xinc, yinc,
                              yflip, rotation, mapbasj, nmap);


    if (itop <= ibas){
        ii1 = itop;
        ii2 = ibas;
    }
    else {
        ii1 = ibas;
        ii2 = itop;
    }

    /* extend with one to avoid edge effects missing values */
    if (ii1 > 1) ii1--;
    if (ii2 < mcol) ii2++;

    if (jtop <= jbas){
        jj1 = jtop;
        jj2 = jbas;
    }
    else {
        jj1 = jbas;
        jj2 = jtop;
    }

    /* extend with one to avoid edge effects missing values */
    if (jj1 > 1) jj1--;
    if (jj2 < mrow) jj2++;

    *i1 = ii1;
    *i2 = ii2;
    *j1 = jj1;
    *j2 = jj2;

}


long _grd3d_point_in_cell(int ic, int jc, double xc, double yc,
                          double zc, int nx, int ny, int   nz,
                          double *p_coor_v, double *p_zcorn_v)

{
    /*
     * This is a special version of the finding a point in cell, optimised
     * for this work flow. The search will basically be in vertical direction.
     *
     * ic, jc        Proposed cell I J (estimated by previous steps in work flow)
     * xc, yc, zc    Point to evaluate if inside cell
     * nx, ny, nz    Dimensions
     * p_coor_v      Coordinates (grid)
     * p_zcorn_v     ZCORN (grid)
     */


    int k;
    long ib;

    for (k = 1; k <= nz; k++) {
        ib = x_ijk2ib(ic, jc, k, nx, ny, nz, 0);
        /* get the corner for the cell */
        double corners[24];
        grd3d_corners(ic, jc, k, nx, ny, nz, p_coor_v, p_zcorn_v,
                      corners, XTGDEBUG);

        if (x_chk_point_in_cell(xc, yc, zc, corners, 1, XTGDEBUG) > 0) {
            return ib;
        }
    }

    return -1; /* if nothing found */
}


void _point_val_ij(int *ires, int *jres, double xc, double yc, double zc, int nx,
                   int ny, double *p_coor_v, double *p_zcornone_v,
                   int i1, int i2, int j1, int j2)
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
     * p_zcorn_v        Coordinates ZCORN
     * i1, i2, j1, j2   I J search range
     */

    int ii, jj;

    *ires = UNDEF_INT;
    *jres = UNDEF_INT;

    for (jj = j1; jj <= j2; jj++) {
        for (ii = i1; ii <= i2; ii++) {

            long ibfound = _grd3d_point_in_cell(ii, jj, xc, yc, zc, nx, ny, 1,
                                                p_coor_v, p_zcornone_v);
            if (ibfound >= 0) {
                int kres;
                x_ib2ijk(ibfound, ires, jres, &kres, nx, ny, 1, 0);
                return;
            }
        }
    }
}

/*
****************************************************************************************
* public function
****************************************************************************************
*/

int grd3d_points_ijk_cells(
    double *xvec,
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
    double *p_zcorn_v,
    int *p_actnum_v,
    double *p_val_v,
    double *p_zcornone_v,
    int *p_actnumone_v,

    int *ivec,
    long nivec,
    int *jvec,
    long njvec,
    int *kvec,
    long nkvec
    )
{

    logger_init(__FILE__, __FUNCTION__);

    logger_info(__LINE__, "Entering routine %s", __FUNCTION__);

    if (nxvec != nyvec || nyvec != nzvec) logger_critical(__LINE__, "Input bug");

    long ib = 0;

    long ibs1 = -1;
    long ibs2 = -1;

    int k1 = 1;
    int k2 = nz;

    int ic;
    for (ic = 0; ic < nxvec; ic++) {
        double xc = xvec[ic];
        double yc = yvec[ic];
        double zc = zvec[ic];

        /*
         * first get an approximate I and J range based on these maps
         * This is based on xc, yc postion in the regular map
         */

        int i1, i2, j1, j2;
        _get_ij_range2(&i1, &i2, &j1, &j2, xc, yc, mcol, mrow, xori, yori, xinc, yinc,
                       yflip, rotation, maptopi, maptopj, mapbasi, mapbasj);

        /*
         * next check the onelayer version of the grid first (speed up)
         * This should pin I J coordinate
         */

        int ires, jres;
        _point_val_ij(&ires, &jres, xc, yc, zc, nx, ny, p_coor_v, p_zcornone_v,
                      i1, i2, j1, j2);

        ivec[ic] = ires;
        jvec[ic] = jres;
        kvec[ic] = UNDEF_INT;

        if (ires < UNDEF_INT_LIMIT) {

            /*
             * means that the  X Y Z point is somewhere inside
             * so now it is time to find exact K location
             */

            long ibfound = _grd3d_point_in_cell(ires, jres, xc, yc, zc, nx, ny, nz,
                                                p_coor_v, p_zcorn_v);
            if (ibfound >= 0) {
                int kres;
                x_ib2ijk(ibfound, &ires, &jres, &kres, nx, ny, nz, 0);
                kvec[ic] = kres;
            }
        }
    }

    logger_info(__LINE__, "Exit from routine %s", __FUNCTION__);

    return EXIT_SUCCESS;

}
