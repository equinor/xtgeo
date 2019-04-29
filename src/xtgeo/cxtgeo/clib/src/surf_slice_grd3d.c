/*
 ******************************************************************************
 *
 * NAME:
 *    surf_slice_grd3d.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Sample values from grd3d based on map values
 *
 * ARGUMENTS:
 *    mcol, mrow     i     map dimens
 *    xori ... yinc  i     Various map settings
 *    p_slice_v      i     map array, e.g a FWL
 *    p_map_v        o     map array, to update to output
 *    ncol, .. nlay  i     Grid dimensions I J K
 *    p_coord_v      i     Grid COORD
 *    p_zcorn_v      i     Grid Z corners for input
 *    p_actnum_v     i     Grid ACTNUM parameter input
 *    p_prop_v       i     Grid property to extract values for
 *    buffer         i     A buffer number of nodes to extend sampling
 *    iflag          i     Options flag
 *    debug          i     Debug level
 *
 * RETURNS:
 *    The C macro EXIT_SUCCESS unless problems + changed pointers
 *
 * TODO/ISSUES/BUGS:
 *    Code is not finished
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ******************************************************************************
 */


#include "libxtg.h"
#include "libxtg_.h"


int surf_slice_grd3d (int mcol,
                      int mrow,
                      double xori,
                      double xinc,
                      double yori,
                      double yinc,
                      double rotation,
                      int yflip,
                      double *p_slice_v,  /* input */
                      long mslice,
                      double *p_map_v,    /* output */
                      long mmap,
                      int ncol,
                      int nrow,
                      int nlay,
                      double *p_coord_v,
                      double *p_zcorn_v,
                      int *p_actnum_v,
                      double *p_prop_v,
                      int buffer,
                      int option,
                      int debug
                      )
{
    /* locals */
    char sbn[24] = "surf_slice_grd3d";
    int j, k, kc1, kc2, kstep = 0, ier, ier3, ios, ix;
    int imm, im, jm, im1, im2, jm1, jm2;
    double corners[24];
    double rx, ry, cellvalue, xm, ym, zm, zgrdtop, zgrdbot;
    double zmapmin, zmapmax;
    double xc[8], yc[8];
    long ib, ic, nactive=0;

    xtgverbose(debug);
    xtg_speak(sbn, 2, "Running %s", sbn);

    /* determine Z window for map (could speed up if flat OWC contact) */
    ier = surf_zminmax(mcol, mrow, p_slice_v, &zmapmin, &zmapmax, debug);
    if (ier == -2) xtg_error(sbn, "Only UNDEF in input map. STOP");

    xtg_speak(sbn, 2, "Map min max %f %f", zmapmin, zmapmax);

    for (ic = 0; ic < mcol*mrow; ic++) p_map_v[ic] = UNDEF;

    /* loop grid3d columns innermost, and find approximate area for map
       to search */

    int IDBG = 0;
    int JDBG = 0;

    for (j = 1; j <= nrow; j++) {

        /* if (j%10 == 0) { */
        /*     printf("Working with row %d of %d\n", j, nrow) ; */
        /* } */

        int i;
        for (i = 1; i <= ncol; i++) {

            /* if the whole column is outside zmap minmax, then skip */
            zgrdtop = grd3d_zminmax(i, j, 1, ncol, nrow, nlay, p_zcorn_v,
                                    0, debug);
            zgrdbot = grd3d_zminmax(i, j, nlay, ncol, nrow, nlay, p_zcorn_v,
                                    1, debug);

            if (zgrdbot < zmapmin) continue;
            if (zgrdtop > zmapmax) continue;

            kc1 = 1;
            kc2 = 0;
            nactive = 0;
            for (k = 1; k <= nlay; k++) {

                ib = x_ijk2ib(i, j, k, ncol, nrow, nlay, 0);
                if (p_actnum_v[ib] == 1) nactive++;

                zgrdtop = grd3d_zminmax(i, j, k, ncol, nrow, nlay,
                                        p_zcorn_v, 0, debug);
                zgrdbot = grd3d_zminmax(i, j, k, ncol, nrow, nlay,
                                        p_zcorn_v, 1, debug);

                if (zgrdbot < zmapmin) kc1 = k;
                if (zgrdtop > zmapmax) {
                    kc2 = k;

                    if (i==IDBG && j==JDBG) printf("ZMAP MIN MAX %f %f\n",
                                                   zmapmin, zmapmax);
                    if (i==IDBG && j==JDBG) printf("Propose K1 K2 %d %d\n",
                                                   kc1, kc2);
                    if (i==IDBG && j==JDBG) printf("K cells: %f %f\n",
                                                   zgrdtop, zgrdbot);
                    break;
                }
            }

            if (nactive == 0) continue;

            if (i==IDBG && j==JDBG) printf("K cells final: %d %d for %d %d\n",
                                           kc1, kc2, i, j);

            if (kc1 > kc2) kc2 = nlay;

            if (i==IDBG && j==JDBG) printf("K cells final: %d %d for %d %d\n",
                                           kc1, kc2, i, j);

            grd3d_corners(i, j, kc1, ncol, nrow, nlay, p_coord_v,
                          p_zcorn_v, corners, debug);
            kstep = 0;
            for (ix = 0; ix < 4; ix++) {
                xc[ix] = corners[ix + kstep];
                yc[ix] = corners[ix + kstep + 1];
                kstep = kstep + 2;
                if (i==IDBG && j==JDBG) printf("Top: IX: XC YC: %d %f %f\n", ix,
                                               xc[ix], yc[ix]);
            }

            grd3d_corners(i, j, kc2, ncol, nrow, nlay, p_coord_v,
                          p_zcorn_v, corners, debug);
            kstep = 8;
            for (ix = 4; ix < 8; ix++) {
                xc[ix] = corners[ix + kstep];
                yc[ix] = corners[ix + kstep + 1];
                kstep = kstep + 2;
                if (i==IDBG && j==JDBG) printf("Bot: IX: XC YC: %d %f %f\n", ix,
                                               xc[ix], yc[ix]);
            }


            /* find widest range in map nodes to cover this cell column
               which will be the upper and lower cell */
            im1 = mcol; im2 = 1; jm1 = mrow; jm2 = 1;

            for (ix = 0; ix < 8; ix++) {
                ier = sucu_ij_from_xy(&im, &jm, &rx, &ry, xc[ix], yc[ix],
                                      xori, xinc, yori, yinc, mcol, mrow,
                                      yflip, rotation, 0, debug);
                if (ier == 0) {
                    if (im < im1) im1 = im;
                    if (im > im2) im2 = im;
                    if (jm < jm1) jm1 = jm;
                    if (jm > jm2) jm2 = jm;
                }

                if (i==IDBG && j==JDBG) printf("IX: XC YC: %d %f %f\n", ix,
                                               xc[ix], yc[ix]);
            }

            /* extend with buffer nodes to be certain */
            im1 -= buffer; im2 += buffer; jm1 -= buffer; jm2 += buffer;
            if (im1 < 1) im1 = 1; if (im2 > mcol) im2 = mcol;
            if (jm1 < 1) jm1 = 1; if (jm2 > mrow) jm2 = mrow;

            if (i==IDBG && j==JDBG) printf("IM: %d %d -- %d %d\n", im1, im2,
                                           jm1, jm2);


            for (k = kc1; k <= kc2; k++) {
                /* get map cell corners: */
                grd3d_corners(i, j, k, ncol, nrow, nlay, p_coord_v,
                              p_zcorn_v, corners, debug);

                ib = x_ijk2ib(i, j, k, ncol, nrow, nlay, 0);
                if (p_actnum_v[ib] == 1) {
                    cellvalue = p_prop_v[ib];
                }
                else{
                    continue;
                }

                for (im = im1; im <= im2; im++) {
                    for (jm = jm1; jm <= jm2; jm++) {
                        ier3 = surf_xyz_from_ij(im, jm, &xm, &ym, &zm,
                                                xori, xinc, yori,
                                                yinc, mcol, mrow,
                                                yflip, rotation,
                                                p_slice_v, mslice,
                                                0, debug);

                        if (ier3 == 0 && zm < UNDEF_LIMIT) {

                            ios = x_chk_point_in_cell(xm, ym, zm,
                                                          corners, 0, debug);

                            if (ios > 0) {
                                imm = x_ijk2ic(im, jm, 1, mcol, mrow, 1, 0);
                                p_map_v[imm] = cellvalue;
                                if (i==IDBG && j==JDBG) printf("Found! %f\n",
                                                               cellvalue);
                            }
                        }
                    }
                }
            }
        }
    }

    return EXIT_SUCCESS;

}
