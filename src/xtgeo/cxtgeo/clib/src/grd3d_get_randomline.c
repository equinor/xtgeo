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
                        double yflip,
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

                        double *values,
                        long nvalues,
                        int option,
                        int debug
                        )
{
    /* locals */
    char sbn[24] = "grd3d_get_randomline";
    int  ib, ic, izc, ier, i1, i2, j1, j2, k1, k2;
    float val, zsam, xc, yc, zc;
    long nmap = mcol * mrow;

    xtgverbose(debug);

    if (debug > 2) xtg_speak(sbn, 3, "Entering routine %s", sbn);

    /* make a 1 K cell grid for simple scan */
    p_z1_v = calloc(8 * nx *ny, sizeof(double));
    p_a1_v = calloc(nx *ny, sizeof(int));

    ier = grd3d_reduce_onelayer(nx, ny, nz, p_zcorn_v, p_z1_v,
                                p_actnum_v, p_a1_v, &nact, 0, debug);

    zsam = (zmax - zmin) / (nzsam - 1);

    ib = -1;
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

        for (izc = 0; izc < nzsam; izc++) {

            zc = zmin + izc * zsam;

            ier = grd3d_point_val_crange(ib, xc, yc, zc, nx, ny, nz, p_coor_v,
                                         p_corn_v, p_actnum_v, pbal_v, &value,
                                         i1, i2, j1, j2, 1, nz, &ibs, 0, debug);

            if (ier == 0) values[ib] = val;
            if (ier != 0) values[ib] = UNDEF;

            ib++;
        }
    }

    if (nvalues != ib) {
        xtg_error(sbn, "Bug probably in %s", sbn);
        return -1;
    }

    return EXIT_SUCCESS;

}
