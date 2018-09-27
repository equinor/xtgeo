/*
 ******************************************************************************
 *
 * NAME:
 *    surf_slice_cube_window.c  IN PROGRESS
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *     Given a map and a cube, sample a series if cube values based on maps
 *     which has increment in Z; then several attributes are computed such
 *     as min/max/avg etc over a vertical window.
 *
 * ARGUMENTS:
 *    ncx...ncz      i     cube dimensions
 *    cxori...czinc  i     cube origin + increment in xyz
 *    crotation      i     Cube rotation
 *    yflip          i     Cube YFLIP index
 *    p_cubeval_v    i     1D Array of cube values of ncx*ncy*ncz size
 *    ncube          i     Length of cube array
 *    mx, my         i     Map dimensions
 *    xori...        i     Map origin, incs, rotation
 *    p_map_v        i     Input map array with Z values
 *    nmap           i     Length of slice array (mx * my)
 *    zincr          i     Z increment
 *    nzincr         i     number of Z increments
 *    p_map_v        o     Maps to update. Allocated as nmap = mx * my * nattr
 *                         where nattr is number of map attributes
 *    nmap           i     Length of output map array (pre allocated)
 *    nattr          i     Number of attributes; these are fixed entries:
 *                         1. MIN; 2. MAX, 3. AVG, 4. STD, 5. RMS
 *    option1        i     Options:
 *                         0: use cube cell value (no interpolation;
 *                            nearest node)
 *                         1: trilinear interpolation in cube
 *                         2: trilinear interpolation and snap to closest X Y
 *    option2        i     0: Leave surf undef if outside cube
 *                         1: Keep surface values as is outside cube
 *    debug          i     Debug level
 *
 * RETURNS:
 *    Function: 0: upon success. If problems <> 0: TODO: UPDATE LIST
 *             -5: No map values sampled
 *             -4: More than 1 sample but less than 10% of map values sampled
 *             -9: Fail in cube_value_ijk (unexpected error)
 *    map pointers updated
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    See XTGeo lisence
 *
 ******************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"
#include <math.h>
#include <time.h>


void _compute_attrs(double *tmpzval, int nzval, double *zattrv, int nattr) {

    /* tmpzval: stack of zvalues from nzval slices */
    /* zattrv: attribute result value, for nattr attributes */

    int icn, iat;

    int nact = 0;
    double zmin = UNDEF;
    double zmax = -1 * UNDEF;
    double zsum = 0.0;
    double z2sum = 0.0;

    for (icn = 0; icn < nzval; icn++) {
        if (tmpzval[icn] < UNDEF_LIMIT) {
            nact++;
            zsum = zsum + tmpzval[icn];
            z2sum = z2sum + pow(tmpzval[icn], 2);
            if (tmpzval[icn] < zmin) zmin = tmpzval[icn];
            if (tmpzval[icn] > zmax) zmax = tmpzval[icn];
        }
    }

    if ((zmax - (-1 * UNDEF)) < FLOATEPS) zmax = UNDEF;

    if (nact == 0) {
        for (iat = 0; iat < nattr; iat++) zattrv[iat] = UNDEF;
        return;
    }
    else{
        zattrv[0] = zmin;
        zattrv[1] = zmax;
        zattrv[2] = zsum / nact;
        zattrv[3] = zsum / nact;  // shall be std
        zattrv[4] = sqrt(z2sum / nact);  // rms
    }
}


int surf_slice_cube_window(
                           int ncx,
                           int ncy,
                           int ncz,
                           double cxori,
                           double cxinc,
                           double cyori,
                           double cyinc,
                           double czori,
                           double czinc,
                           double crotation,
                           int yflip,
                           float *p_cubeval_v,
                           long ncube,
                           int mx,
                           int my,
                           double xori,
                           double xinc,
                           double yori,
                           double yinc,
                           int mapflip,
                           double mrotation,
                           double *p_map_v,
                           long nmap,
                           double zincr,
                           int nzincr,
                           double *p_attrs_v,
                           long nattrmaps,
                           int nattr,
                           int option1,
                           int option2,
                           int debug
                           )

{
    /* locals */
    char s[24] = "surf_slice_cube_window";
    int im, jm, knum, ier, iat, ic;
    double xcor, ycor, zcor, zval;
    float value;
    double tmpzval[nzincr];
    double zattr[nattr];
    int option1a = 0;
    time_t t1, t2;
    double elapsed;

    xtgverbose(debug);
    xtg_speak(s, 2, "Entering routine %s", s);

    xtg_speak(s, 1, "Working with slice...");

    time(&t1);
    /* work with every map node */
    for (im = 1; im <= mx; im++) {
        if (debug > 0 && im % 10 == 0) {
            time(&t2);
            elapsed = difftime(t2, t1);
            xtg_speak(s, 1, "Working with map column %d of %d ...(%6.2lf)",
                      im, mx, elapsed);
            time(&t1);
        }

        for (jm = 1; jm <= my; jm++) {

            /* get the surface x, y, value (z) from IJ location */
            ier = surf_xyz_from_ij(im, jm, &xcor, &ycor, &zcor, xori, xinc,
                                   yori, yinc, mx, my, mapflip,
                                   mrotation, p_map_v, nmap, 0, debug);


            if (zcor < UNDEF_LIMIT) {

                /* work with Z increment */
                for (knum = 0; knum < nzincr; knum++) {
                    zval = zcor + knum * zincr;


                    if (option1 == 0) {

                        ier = cube_value_xyz_cell
                            (xcor, ycor, zval, cxori, cxinc, cyori,
                             cyinc, czori, czinc, crotation,
                             yflip, ncx, ncy, ncz,
                             p_cubeval_v, &value, 0,
                             debug);
                    }
                    else if (option1 == 1 || option1 == 2) {

                        option1a = 0;
                        if (option1 == 2) option1a = 1;  // snap to closest XY
                        if (knum > 0) option1a += 10;    // Skip IJ calculation

                        /* TIDSTYV! */
                        ier = cube_value_xyz_interp
                            (xcor, ycor, zval, cxori, cxinc, cyori,
                             cyinc, czori, czinc, crotation,
                             yflip, ncx, ncy, ncz,
                             p_cubeval_v, &value, option1a,
                             debug);


                    }
                    else{
                        xtg_error(s, "Invalid option1 (%d) to %s", option1, s);
                    }


                    if (ier == EXIT_SUCCESS) {
                        tmpzval[knum] = value;
                    }
                    else if (ier == -1 && option2 == 0) {
                        tmpzval[knum] = UNDEF_MAP;
                    }
                    if (zval > UNDEF_LIMIT) tmpzval[knum] = UNDEF_MAP;

                }

                _compute_attrs(tmpzval, nzincr, zattr, nattr);

                /* update attribute for that particular Z vector */
                for (iat = 0; iat < nattr; iat++) {
                    ic = x_ijk2ib(im, jm, iat + 1, mx, my, nattr, 0);
                    p_attrs_v[ic] = zattr[iat];
                }
            /* } */
            /* else{ */
            /*     /\* maps is undefined *\/ */
            /*     for (iat = 0; iat < nattr; iat++) { */
            /*         ic = x_ijk2ib(im, jm, iat + 1, mx, my, nattr, 0); */
            /*         p_attrs_v[ic] = UNDEF; */
                /* } */
            }
        }
    }
    xtg_speak(s, 1, "Working with slices... DONE!");

    return EXIT_SUCCESS;
}
