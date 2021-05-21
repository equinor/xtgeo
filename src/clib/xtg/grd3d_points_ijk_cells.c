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

#include "libxtg.h"
#include "libxtg_.h"
#include "logger.h"

#include <math.h>

/*
****************************************************************************************
* private functions
****************************************************************************************
*/

static int
_get_ij_range(int *i1,
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
    double itop, jtop, ibas, jbas;
    int ii1, ii2, jj1, jj2;

    if (nx < 4 || ny < 4) {
        *i1 = 1;
        *i2 = nx;
        *j1 = 1;
        *j2 = ny;
        return 1;
    }

    nmap = mcol * mrow;

    /* get map value for I J from x y */
    int opt = 2; /* nearest sampling */
    itop = surf_get_z_from_xy(xc, yc, mcol, mrow, xori, yori, xinc, yinc, yflip,
                              rotation, maptopi, nmap, opt);
    jtop = surf_get_z_from_xy(xc, yc, mcol, mrow, xori, yori, xinc, yinc, yflip,
                              rotation, maptopj, nmap, opt);
    ibas = surf_get_z_from_xy(xc, yc, mcol, mrow, xori, yori, xinc, yinc, yflip,
                              rotation, mapbasi, nmap, opt);
    jbas = surf_get_z_from_xy(xc, yc, mcol, mrow, xori, yori, xinc, yinc, yflip,
                              rotation, mapbasj, nmap, opt);

    if (itop >= UNDEF_LIMIT || jtop >= UNDEF_LIMIT || ibas >= UNDEF_LIMIT ||
        jbas >= UNDEF_LIMIT) {
        return -1;
    }

    if (itop <= ibas) {
        ii1 = (int)itop;
        ii2 = (int)ibas;
    } else {
        ii1 = (int)ibas;
        ii2 = (int)itop;
    }

    /* extend with one to avoid edge effects of missing values */
    if (ii1 > 1)
        ii1--;
    if (ii2 < mcol)
        ii2++;

    if (jtop <= jbas) {
        jj1 = (int)jtop;
        jj2 = (int)jbas;
    } else {
        jj1 = (int)jbas;
        jj2 = (int)jtop;
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

    return 1;
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
                     double *coordsv,
                     double *zcornsv,
                     int *score)

{
    /*
     * This is a special version of the finding a point in cell, optimised
     * for this work flow.
     *
     * ic, jc, kc    Proposed cell I J (estimated by previous steps in work flow)
     * xc, yc, zc    Point to evaluate if inside cell
     * nx, ny, nz    Dimensions
     * coordsv       Pillar coordinates (grid)
     * zcornsv       ZCORN (grid)
     * score         This is a number telling how good the match is, going from -1 to 24
     */

    long ib;

    ib = x_ijk2ib(ic, jc, kc, nx, ny, nz, 0);

    /* get the corner for the cell */
    double corners[24];
    grd3d_corners(ic, jc, kc, nx, ny, nz, coordsv, 0, zcornsv, 0, corners);

    *score = x_point_in_hexahedron(xc, yc, zc, corners, 24, 1);

    if (*score > 0) {
        return ib;
    }

    return -1; /* if nothing found */
}

static int
_point_val_ij(double xc,
              double yc,
              double zc,
              int nx,
              int ny,
              double *coordsv,
              double *p_zcornone_v,
              int i1,
              int i2,
              int j1,
              int j2,
              long ibfound[])
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
     * coordsv         Coordinates COORD
     * zcornsv          Coordinates ZCORN
     * ibalts           It may be that several IB ranges may be valid
     * i1, i2, j1, j2   I J search range
     */

    int ii, jj;

    int score;

    int ibn = 0;

    for (jj = j1; jj <= j2; jj++) {
        for (ii = i1; ii <= i2; ii++) {

            score = 0;
            long ibfoundp = _grd3d_point_in_cell(ii, jj, 1, xc, yc, zc, nx, ny, 1,
                                                 coordsv, p_zcornone_v, &score);

            if (score > 50) {
                ibfound[ibn++] = ibfoundp;
                return ibn;
            } else if (score == 50) {
                ibfound[ibn++] = ibfoundp;
                if (ibn == 4)
                    return ibn;
            }
        }
    }
    return ibn;
}

static int
_point_val_ijk(double xc,
               double yc,
               double zc,
               int nx,
               int ny,
               int nz,
               double *coordsv,
               double *zcornsv,
               int *actnumsv,
               int actnumoption,
               int iin,
               int jin,
               long *ibchosen)
{
    /*
     * The purpose here is to search the final layer grid for K location of point XYZ
     *
     * xc, yc, zc       Points to evaluate if inside
     * nx, ny, nz       Dimensions
     * coordsv         Coordinates COORD
     * zcornsv        Coordinates ZCORN
     * iin, jin         I J column
     */

    int score;

    int k, nib;

    long *ibalts = calloc(nz, sizeof(long));
    int *ibscore = calloc(nz, sizeof(int));
    int *ibactive = calloc(nz, sizeof(int));

    nib = 0;
    for (k = 1; k <= nz; k++) {
        long ibfound = _grd3d_point_in_cell(iin, jin, k, xc, yc, zc, nx, ny, nz,
                                            coordsv, zcornsv, &score);

        if (score >= 50) {
            ibalts[nib] = ibfound;
            ibscore[nib] = score;
            ibactive[nib] = actnumsv[ibfound];
            if (actnumoption == 0)
                ibactive[nib] = 1;  // appear active

            logger_debug(LI, FI, FU,
                         "ZC = %6.2lf: I J K = %d %d %d (IB = %ld), score = %d", zc,
                         iin, jin, k, ibalts[nib], ibscore[nib]);

            nib++;
        } else if (score == 0 && nib > 0) {
            break;
        }
    }

    *ibchosen = -1;

    int retvalue = -1;

    if (nib > 0) {
        int n;
        int hiscore = 0;
        for (n = 0; n < nib; n++) {
            if (ibscore[n] > hiscore && ibactive[n] == 1) {
                hiscore = ibscore[n];
                *ibchosen = ibalts[n];
                logger_debug(LI, FI, FU, "IBCHOSEN is %ld with score %d", *ibchosen,
                             hiscore);
            }
        }
        retvalue = hiscore;
    }

    free(ibalts);
    free(ibscore);
    free(ibactive);

    logger_debug(LI, FI, FU, "Final IBCHOSEN is %ld with retvalue %d", *ibchosen,
                 retvalue);
    return retvalue;
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

                       double *coordsv,
                       long ncoordin,
                       double *zcornsv,
                       long nzcornin,
                       int *actnumsv,
                       long nactin,

                       double *p_zcornone_v,
                       long nzcornonein,

                       int actnumoption,

                       int *ivec,
                       long nivec,
                       int *jvec,
                       long njvec,
                       int *kvec,
                       long nkvec)
{

    logger_info(LI, FI, FU, "Entering routine %s", FU);

    if (nxvec != nyvec || nyvec != nzvec) {
        memset(ivec, 0, sizeof(int) * nivec);
        memset(jvec, 0, sizeof(int) * njvec);
        memset(kvec, 0, sizeof(int) * nkvec);
        throw_exception("nxvec != nyvec or nyvec != nzvec in: grd3d_points_ijk_cells");
        return EXIT_FAILURE;
    }
    if (nivec != njvec || nivec != nkvec) {
        memset(ivec, 0, sizeof(int) * nivec);
        memset(jvec, 0, sizeof(int) * njvec);
        memset(kvec, 0, sizeof(int) * nkvec);
        throw_exception("nivec != njvec or nivec != nkvec in: grd3d_points_ijk_cells");
        return EXIT_FAILURE;
    }
    int ic;
    for (ic = 0; ic < nxvec; ic++) {
        double xc = xvec[ic];
        double yc = yvec[ic];
        double zc = zvec[ic];

        ivec[ic] = UNDEF_INT;
        jvec[ic] = UNDEF_INT;
        kvec[ic] = UNDEF_INT;

        /*
         * first get an approximate I and J range based on these maps
         * This is based on xc, yc position in the regular map
         */

        int i1, i2, j1, j2;
        int ier =
          _get_ij_range(&i1, &i2, &j1, &j2, xc, yc, mcol, mrow, xori, yori, xinc, yinc,
                        yflip, rotation, maptopi, maptopj, mapbasi, mapbasj, nx, ny);

        if (ier < 0)
            continue;

        /*
         * next check the onelayer version of the grid first (speed up)
         * This should pin I J coordinate
         */
        long ibfound[4];
        int nfound = _point_val_ij(xc, yc, zc, nx, ny, coordsv, p_zcornone_v, i1, i2,
                                   j1, j2, ibfound);

        if (nfound > 0) {

            int ibn;
            for (ibn = 0; ibn < nfound; ibn++) {

                /*
                 * means that the  X Y Z point is somewhere inside
                 * so now it is time to find exact K location
                 */

                int ires, jres, kres;
                x_ib2ijk(ibfound[ibn], &ires, &jres, &kres, nx, ny, 1, 0);

                logger_debug(LI, FI, FU,
                             "Onelayer hit found for point %.3lf %.3lf %.3lf"
                             " for column I J = %d %d",
                             xc, yc, zc, ires, jres);

                long ibfound2;
                int nscore =
                  _point_val_ijk(xc, yc, zc, nx, ny, nz, coordsv, zcornsv, actnumsv,
                                 actnumoption, ires, jres, &ibfound2);
                if (ibfound2 >= 0 && nscore > 0) {

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
                    break;
                }
            }
        }
    }

    logger_info(LI, FI, FU, "Exit from routine %s", FU);

    return EXIT_SUCCESS;
}
