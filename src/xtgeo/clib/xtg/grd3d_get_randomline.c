/*
 ***************************************************************************************
 *
 * NAME:
 *    grd3d_get_randomline.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Given X Y Z vectors, return a a randomline array from a 3D grid property
 *
 * ARGUMENTS:
 *    xvec, yvec          i     Arrays coords XY
 *    zmin, zmax          i     Vertical range
 *    nzsam               i     Vertical sampling numbering
 *    mcol, mrow          i     Number of rows/cols for maps
 *    xori..rotation      i     Map settings
 *    maptopi..mapbasj    i     Map arrays for I J top/base
 *    nx ny nz            i     Grid dimensions
 *    p_zcorn_v           i     Grid Zcorn
 *    p_coord_v           i     Grid ZCORN
 *    p_acnum_v           i     Grid ACTNUM
 *    p_val_v             i     3D Grid values
 *    p_zcornone_v        i     Grid ZCORN
 *    p_acnumone_v        i     Grid ACTNUM
 *    value               o     Randomline array
 *    option              i     For later
 *    debug               i     Debug level
 *
 * RETURNS:
 *    Array length, -1 if fail
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
* private function
****************************************************************************************
*/

void _get_ij_range(int *i1,  int *i2, int *j1, int *j2, double xc, double yc, int mcol,
                   int mrow, double xori, double yori, double xinc, double yinc,
                   int yflip, double rotation, double *maptopi, double *maptopj,
                   double *mapbasi, double *mapbasj, int nx, int ny)
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

    /*  if numbers are unphysical for some reason, revert to grid limits */
    if (ii1 < 1 || ii1 >= nx) ii1 = 1;
    if (ii2 <= 1 || ii2 > nx) ii2 = nx;
    if (jj1 < 1 || jj1 >= ny) jj1 = 1;
    if (jj2 <= 1 || jj2 > ny) jj2 = ny;

    if (ii2 <= ii1 || (ii2 - ii1) >= nx || jj2 <= jj1 || (jj2 - jj1) >= ny) {
        ii1 = 1; ii2 = nx; jj1 = 1; jj2 = ny;
    }

    *i1 = ii1;
    *i2 = ii2;
    *j1 = jj1;
    *j2 = jj2;

}

/*
****************************************************************************************
* public function
****************************************************************************************
*/

int grd3d_get_randomline(
                         double *xvec,
                         long nxvec,
                         double *yvec,
                         long nyvec,
                         double zmin,
                         double zmax,
                         int nzsam,

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

                         double *values,
                         long nvalues,

                         int option
                         )
{
    /* locals */
    int  ib, ic, izc, ier, ios, i1, i2, j1, j2, k1, k2;
    long ibs1, ibs2;
    double zsam;
    double value, *p_dummy_v = NULL;

    logger_info(LI, FI, FU, "Entering routine %s", FU);

    zsam = (zmax - zmin) / (nzsam - 1);

    if (nxvec != nyvec || nxvec != nvalues) {
        logger_warn(LI, FI, FU, "There seems to be issues in %s:"
                    "NXVEC = %ld, NYVEC = %ld, NVALUES = %ld", FU,
                    nxvec, nyvec, nvalues);
    }

    ib = 0;

    ibs1 = -1;
    ibs2 = -1;

    k1 = 1;
    k2 = nz;

    for (ic = 0; ic < nxvec; ic++) {
        double xc = xvec[ic];
        double yc = yvec[ic];

        _get_ij_range(&i1, &i2, &j1, &j2, xc, yc, mcol, mrow, xori, yori, xinc, yinc,
                      yflip, rotation, maptopi, maptopj, mapbasi, mapbasj, nx, ny);

        for (izc = 0; izc < nzsam; izc++) {

            double zc = zmin + izc * zsam;

            /* check the onelayer version of the grid first (speed up) */
            ier = grd3d_point_val_crange(xc, yc, zc, nx, ny, 1, p_coor_v,
                                         p_zcornone_v, p_actnumone_v, p_dummy_v, &value,
                                         i1, i2, j1, j2, 1, 1, &ibs1, -1, XTGDEBUG);


            if (ier == 0) {


                ios = grd3d_point_val_crange(xc, yc, zc, nx, ny, nz, p_coor_v,
                                             p_zcorn_v, p_actnum_v, p_val_v,
                                             &value, i1, i2, j1, j2, k1, k2,
                                             &ibs2, 0, XTGDEBUG);

                if (ios == 0) {
                    values[ib++] = value;
                }
                else{
                    values[ib++] = UNDEF;
                }

            }
            else{
                /* outside onelayer cell */
                values[ib++] = UNDEF;
                continue;
            }
        }
    }

    logger_info(LI, FI, FU, "Exit from routine %s", FU);

    return EXIT_SUCCESS;

}
