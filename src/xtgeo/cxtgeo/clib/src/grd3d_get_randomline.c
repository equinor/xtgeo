/*
 ******************************************************************************
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
 *    x, y           i     Arrays coords XY
 *    zmin, zmax     i     Vertical range
 *    zsampling      i     vertical sampling increment
 *    nx ny nz       i     Grid dimensions
 *    p_zcorn_v      i     Grid Zcorn
 *    p_coord_v      i     Grid ZCORN
 *    p_acnum_v      i     Grid ACTNUM
 *    p_val_v        i     3D Grid values
 *    value          o     Randomline array
 *    option         i     For later
 *    debug          i     Debug level
 *
 * RETURNS:
 *    Array length, -1 if fail
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ******************************************************************************
 */


#include "libxtg.h"
#include "libxtg_.h"

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

                         int option,
                         int debug
                         )
{
    /* locals */
    char sbn[24] = "grd3d_get_randomline";
    int  ib, ic, izc, ier, ios, i1, i2, j1, j2, k1, k2, itop, jtop, ibas, jbas;
    long ibs1, ibs2;
    double val, zsam, xc, yc, zc;
    long nmap = mcol * mrow;
    double value, *p_dummy_v;

    debug = 1;
    xtgverbose(debug);

    if (debug > 2) xtg_speak(sbn, 3, "Entering routine %s", sbn);

    zsam = (zmax - zmin) / (nzsam - 1);

    p_dummy_v = calloc(nx * ny, sizeof(float));

    ib = 0;

    ibs1 = -1;
    ibs2 = -1;

    k1 = 1;
    k2 = nz;

    for (ic = 0; ic < nxvec; ic++) {
        xc = xvec[ic];
        yc = yvec[ic];

        /* get map value for I J from x y */
        itop = surf_get_z_from_xy(xc, yc, mcol, mrow, xori, yori, xinc, yinc,
                                  yflip, rotation, maptopi, nmap, debug);
        jtop = surf_get_z_from_xy(xc, yc, mcol, mrow, xori, yori, xinc, yinc,
                                  yflip, rotation, maptopj, nmap, debug);
        ibas = surf_get_z_from_xy(xc, yc, mcol, mrow, xori, yori, xinc, yinc,
                                  yflip, rotation, mapbasi, nmap, debug);
        jbas = surf_get_z_from_xy(xc, yc, mcol, mrow, xori, yori, xinc, yinc,
                                  yflip, rotation, mapbasj, nmap, debug);

        if (itop <= ibas){i1 = itop; i2 = ibas;} else {i1 = ibas; i2 = itop;}
        if (jtop <= jbas){j1 = jtop; j2 = jbas;} else {j1 = jbas; j2 = jtop;}

        if (debug > 0) xtg_speak(sbn, 1, "I J range %d %d %d %d...", i1, i2, j1, j2);

        for (izc = 0; izc < nzsam; izc++) {

            zc = zmin + izc * zsam;

            /* check the onelayer version of the grid first */
            ier = grd3d_point_val_crange(xc, yc, zc, nx, ny, 1, p_coor_v,
                                         p_zcornone_v, p_actnumone_v, p_dummy_v, &value,
                                         i1, i2, j1, j2, 1, 1, &ibs1, 0, debug);

            if (debug > 0 && ier == 0) xtg_speak(sbn, 1, "Trying sample %f %f %f onelayer: %d",
                                     xc, yc, zc, ier);

            if (ier == -1) {
                continue;
            }
            else if (ier == 0) {

                if (debug > 0 && ier == 0) xtg_speak(sbn, 1, "Trying K1 K2 %d %d",
                                                     k1, k2);

                ios = grd3d_point_val_crange(xc, yc, zc, nx, ny, nz, p_coor_v,
                                             p_zcorn_v, p_actnum_v, p_val_v,
                                             &value, i1, i2, j1, j2, k1, k2,
                                             &ibs2, 0, debug);

                if (ios == 0) {
                    values[ib] = value;
                }
                else{
                    values[ib] = UNDEF;
                }
                ib++;
            }
        }
    }

    if (debug > 2) xtg_speak(sbn, 3, "Done ...");
    /* if (nvalues != ib) { */
    /*     xtg_error(sbn, "Bug probably in %s", sbn); */
    /*     return -1; */
    /* } */

    return EXIT_SUCCESS;

}
