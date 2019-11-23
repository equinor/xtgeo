/* test the surf_* routines */

#include "../tap/tap.h"
#include "../src/libxtg.h"
#include "../src/libxtg_.h"
#include <math.h>
#include <stdio.h>

int aresame(double a, double b);

int aresame(double a, double b)
{
    if (fabs(a - b) < FLOATEPS) return 1;
    return 0;
}

int main () {
    int    ib, ier, option=0, i, j, i1, j1, i2, j2, i3, j3, i4, j4;
    char   file[70];
    int    mx, my;
    double xori, yori, xinc, yinc, rot, xx, yy, zz;
    int    mode, debug, yflip, mflip;
    long   ndef, nsurf, nn;
    double *p_map_v, *p_x_v, *p_y_v, diff, *xxmap, rx, ry, rx1, ry1, rx2, ry2;
    double rx3, ry3, rx4, ry4;

    debug = 1;

    xtgverbose(debug);
    xtg_verbose_file("NONE");


    plan(NO_PLAN);


    /*
     * -------------------------------------------------------------------------
     * Some theoritical test first
     * -------------------------------------------------------------------------
     */

    /* xori = 0.0; */
    /* yori = 0.0; */
    /* xinc = 10.0; */
    /* yinc = 10.0; */
    /* mx = 10; */
    /* my = 10; */
    /* rot = 0.0; */
    /* yflip = 1; */
    /* xx = 59; */
    /* yy = 50; */
    /* option = 1; */

    /* ier = sucu_ij_from_xy(&i1, &j1, &rx1, &ry1, xx, yy, xori, xinc, yori, yinc, */
    /*     		  mx, my, yflip, rot, option, debug); */

    /* option = 0; */
    /* ier = sucu_ij_from_xy(&i2, &j2, &rx2, &ry2, xx, yy, xori, xinc, yori, yinc, */
    /*     		  mx, my, yflip, rot, option, debug); */


    /* ok(i1 == i2 - 1, "Shall be the shifted for I"); */
    /* ok(j1 == j2, "Shall be same for J"); */


    /* /\* do flipping --------------------------------------------------------*\/ */

    /* yflip = -1; */
    /* yy = -50; */
    /* option = 1; */
    /* ier = sucu_ij_from_xy(&i3, &j3, &rx3, &ry3, xx, yy, xori, xinc, yori, yinc, */
    /*     		  mx, my, yflip, rot, option, debug); */

    /* option = 0; */
    /* ier = sucu_ij_from_xy(&i2, &j2, &rx2, &ry2, xx, yy, xori, xinc, yori, yinc, */
    /*     		  mx, my, yflip, rot, option, debug); */

    /* ok(i3 == i1, "Same I for flip"); */
    /* ok(ry3 == -1 * ry1, "Flipped sign for RY"); */


    /* /\* other coords *\/ */
    /* xx = 59; */
    /* yy = 59; */

    /* option = 1; */
    /* yflip = 1; */
    /* ier = sucu_ij_from_xy(&i1, &j1, &rx1, &ry1, xx, yy, xori, xinc, yori, yinc, */
    /*     		  mx, my, yflip, rot, option, debug); */
    /* option = 0; */
    /* ier = sucu_ij_from_xy(&i2, &j2, &rx2, &ry2, xx, yy, xori, xinc, yori, yinc, */
    /*     		  mx, my, yflip, rot, option, debug); */
    /* ok (i1 == i2 - 1); */
    /* ok (j1 == j2 - 1); */

    /* yflip = -1; */
    /* yy = -1*yy; */

    /* option = 1; */
    /* ier = sucu_ij_from_xy(&i1, &j1, &rx1, &ry1, xx, yy, xori, xinc, yori, yinc, */
    /*     		  mx, my, yflip, rot, option, debug); */
    /* option = 0; */
    /* ier = sucu_ij_from_xy(&i2, &j2, &rx2, &ry2, xx, yy, xori, xinc, yori, yinc, */
    /*     		  mx, my, yflip, rot, option, debug); */
    /* ok (i1 == i2 - 1); */
    /* ok (j1 == j2 - 1); */




    /* /\* test with angles*\/ */
    /* printf("\nANGLES 30 ------------------------------------\n\n"); */

    /* rot = 30; */
    /* xx = 59; */
    /* yy = 59; */

    /* option = 1; */
    /* yflip = 1; */
    /* ier = sucu_ij_from_xy(&i1, &j1, &rx1, &ry1, xx, yy, xori, xinc, yori, yinc, */
    /*     		  mx, my, yflip, rot, option, debug); */
    /* option = 0; */
    /* ier = sucu_ij_from_xy(&i2, &j2, &rx2, &ry2, xx, yy, xori, xinc, yori, yinc, */
    /*     		  mx, my, yflip, rot, option, debug); */

    /* printf("Flip -1\n"); */
    /* yflip = -1; */
    /* rot = 360 - 30; */
    /* yy = -59; */

    /* option = 1; */
    /* ier = sucu_ij_from_xy(&i3, &j3, &rx3, &ry3, xx, yy, xori, xinc, yori, yinc, */
    /*     		  mx, my, yflip, rot, option, debug); */
    /* option = 0; */
    /* ier = sucu_ij_from_xy(&i4, &j4, &rx4, &ry4, xx, yy, xori, xinc, yori, yinc, */
    /*     		  mx, my, yflip, rot, option, debug); */


    /* ok(aresame(rx1, rx3) == 1, "Angular1 %f %f", rx1, rx3); */
    /* ok(aresame(ry1, -1*ry3) == 1, "Angular2"); */
    /* ok(aresame(rx2, rx4) == 1, "Angular3"); */
    /* ok(aresame(ry2, -1*ry4) == 1, "Angular4"); */

    /* ok(i1 == i3); */
    /* ok(j1 == j3); */
    /* ok(i2 == i2); */
    /* ok(j2 == j4); */


    /* /\* test with other angles*\/ */
    /* printf("\nANGLES 91 ------------------------------------ flag=1 \n\n"); */

    /* rot = 91; */
    /* xx = -20; */
    /* yy = 59; */

    /* option = 1; */
    /* yflip = 1; */
    /* ier = sucu_ij_from_xy(&i1, &j1, &rx1, &ry1, xx, yy, xori, xinc, yori, yinc, */
    /*     		  mx, my, yflip, rot, option, debug); */
    /* option = 0; */
    /* ier = sucu_ij_from_xy(&i2, &j2, &rx2, &ry2, xx, yy, xori, xinc, yori, yinc, */
    /*     		  mx, my, yflip, rot, option, debug); */

    /* printf("Flip -1\n"); */
    /* yflip = -1; */
    /* rot = 360 - 91; */
    /* yy = -59; */

    /* option = 1; */
    /* ier = sucu_ij_from_xy(&i3, &j3, &rx3, &ry3, xx, yy, xori, xinc, yori, yinc, */
    /*     		  mx, my, yflip, rot, option, debug); */
    /* option = 0; */
    /* ier = sucu_ij_from_xy(&i4, &j4, &rx4, &ry4, xx, yy, xori, xinc, yori, yinc, */
    /*     		  mx, my, yflip, rot, option, debug); */


    /* ok(aresame(rx1, rx3) == 1, "Angular1 %f %f", rx1, rx3); */
    /* ok(aresame(ry1, -1*ry3) == 1, "Angular2"); */
    /* ok(aresame(rx2, rx4) == 1, "Angular3"); */
    /* ok(aresame(ry2, -1*ry4) == 1, "Angular4"); */

    /* ok(i1 == i3); */
    /* ok(j1 == j3); */
    /* ok(i2 == i2); */
    /* ok(j2 == j4); */

    /* /\* */
    /*  * ------------------------------------------------------------------------- */
    /*  * Read an existing map in irap binary format (Fossekall) */
    /*  * ------------------------------------------------------------------------- */
    /*  *\/ */

    /* strcpy(file,"../../../xtgeo-testdata/surfaces/fos/1/fossekall1.irapbin"); */

    /* mx = 58; */
    /* my = 53; */
    /* nsurf = mx * my; */
    /* p_map_v = calloc(nsurf,sizeof(double)); */

    /* mode = 0; */
    /* ier = surf_import_irap_bin(file, mode, &mx, &my, &ndef, &xori, &yori, */
    /*     		       &xinc, &yinc, &rot, p_map_v, nsurf, */
    /*     		       option, debug); */


    /* ok(mx == 58, "MX in Irap binary FK file"); */
    /* ok(my == 53, "MY in Irap binary FK file"); */

    /* mode = 1; */
    /* p_map_v = calloc(mx*my,sizeof(double)); */
    /* ier = surf_import_irap_bin(file, mode, &mx, &my, &ndef, &xori, &yori, */
    /*     		       &xinc, &yinc, &rot, p_map_v, nsurf, */
    /*     		       option, debug); */


    /* ok(mx == 58, "MX in Irap binary FK file"); */
    /* ok(my == 53, "MY in Irap binary FK file"); */
    /* ok(ndef == 2072, "NDEF in Irap binary FK file"); */


    /* /\* */
    /*  * ------------------------------------------------------------------------- */
    /*  * Export FK to Irap binary */
    /*  * ------------------------------------------------------------------------- */
    /*  *\/ */

    /* strcpy(file,"TMP/fossekall1_export.irapbin"); */

    /* ier = surf_export_irap_bin(file, mx, my, xori, yori, */
    /*     		       xinc, yinc, rot, p_map_v, nsurf, */
    /*     		       option, debug); */


    /* /\* */
    /*  * ------------------------------------------------------------------------- */
    /*  * Test X Y Z from I J */
    /*  * ------------------------------------------------------------------------- */
    /*  *\/ */

    /* i = 13; */
    /* j = 34; */
    /* mflip = 1;  // yflip for maps */
    /* ier = surf_xyz_from_ij(i, j, &xx, &yy, &zz, xori, xinc, yori, yinc, mx, my, */
    /*                        mflip, rot, p_map_v, mx*my, 0, debug); */

    /* printf("TESTXXXX1: IER  X  Y  Z: %d %f %f %f\n", ier, xx, yy, zz ); */


    /* /\* */
    /*  * ------------------------------------------------------------------------- */
    /*  * Test getting xoordinates of each map node */
    /*  * ------------------------------------------------------------------------- */
    /*  *\/ */
    /* nn = mx * my; */

    /* p_x_v = calloc(nn,sizeof(double)); */
    /* p_y_v = calloc(nn,sizeof(double)); */

    /* ier = surf_xy_as_values(xori, xinc, yori, yinc, mx, my, rot, p_x_v, nn, */
    /*     		    p_y_v, nn, 0, debug); */

    /* i=1; j=1; */
    /* ib=x_ijk2ib(i, j, 1, mx, my, 1,0); */
    /* //printf("Coordinate %d %d has values %f %f\n", i, j, p_x_v[ib], p_y_v[ib]); */
    /* diff = fabs(p_x_v[ib] - 464308.406250); */
    /* ok(diff < 0.01, "Corner XORI coordinate"); */

    /* i=5; j=1; */
    /* ib=x_ijk2ib(i, j, 1, mx, my, 1,0); */
    /* //printf("Coordinate %d %d has values %f %f\n", i, j, p_x_v[ib], p_y_v[ib]); */
    /* diff = fabs(p_x_v[ib] - 464342.608250); */
    /* ok(diff < 0.01, "Corner X 5,1 coordinate"); */

    /* i=1; j=10; */
    /* ib=x_ijk2ib(i, j, 1, mx, my, 1,0); */
    /* //printf("Coordinate %d %d has values %f %f\n", i, j, p_x_v[ib], p_y_v[ib]); */
    /* diff = fabs(p_y_v[ib] - 7337310.45); */
    /* ok(diff < 0.01, "Corner Y 1,10 coordinate"); */


    /* /\* */
    /*  * ------------------------------------------------------------------------- */
    /*  * Test getting distance from point+azimuth as map */
    /*  * Check the maps in RMS... */
    /*  * ------------------------------------------------------------------------- */
    /*  *\/ */
    /* ier = surf_get_dist_values(xori, xinc, yori, yinc, mx, my, rot, */
    /*     		       464960, 7336900, 190, p_map_v, mx*my, 0, debug); */


    /* strcpy(file,"TMP/fossekall1_export_dist.irapbin"); */

    /* ier = surf_export_irap_bin(file, mx, my, xori, yori, */
    /*     		       xinc, yinc, rot, p_map_v, nsurf, */
    /*     		       option, debug); */


    /* /\* */
    /*  * ------------------------------------------------------------------------- */
    /*  * Test get IJ from XY */
    /*  * ------------------------------------------------------------------------- */
    /*  *\/ */

    /* mx = 58; */
    /* my = 53; */
    /* yflip = 1; */

    /* xxmap = calloc(mx*my,sizeof(double)); */

    /* strcpy(file,"../../../xtgeo-testdata/surfaces/fos/1/topreek_rota.gri"); */

    /* mode = 1; */
    /* ier = surf_import_irap_bin(file, mode, &mx, &my, &ndef, &xori, &yori, */
    /*     		       &xinc, &yinc, &rot, xxmap, nsurf, */
    /*     		       option, debug); */

    /* xx = 464388.0; */
    /* yy = 7337196.0; */

    /* ier = sucu_ij_from_xy(&i, &j, &rx, &ry, xx, yy, xori, xinc, yori, yinc, */
    /*     		  mx, my, yflip, rot, 1, debug); */

    /* printf("I J: %d %d  IER=%d\n", i, j, ier); */

    /* ok(i == 3, "Point in cell (node) X try1"); */
    /* ok(j == 3, "Point in cell (node) Y try1"); */

    /* xx = 464388.0; */
    /* yy = 7337196.0; */

    /* ier = sucu_ij_from_xy(&i, &j, &rx, &ry, xx, yy, xori, xinc, yori, yinc, */
    /*                          mx, my, yflip, rot, 1, debug); */

    /* printf("I J: %d %d\n", i, j); */
    /* ok(i == 3, "Point in cell (node) X try1 V2"); */
    /* ok(j == 3, "Point in cell (node) Y try1 V2"); */

    /* xx = 464402.0; */
    /* yy = 7337190.0; */

    /* ier = sucu_ij_from_xy(&i, &j, &rx, &ry, xx, yy, xori, xinc, yori, yinc, */
    /*     		  mx, my, yflip, rot, 1, debug); */


    /* ok(i == 3, "Point in cell (node) X try2"); */
    /* ok(j == 3, "Point in cell (node) Y try2"); */

    /* xx = 464375.992; */
    /* yy = 7337128.076; */

    /* ier = sucu_ij_from_xy(&i, &j, &rx, &ry, xx, yy, xori, xinc, yori, yinc, */
    /*     		  mx, my, yflip, rot, 1, debug); */

    /* ok(i == 5, "Point in cell (node) X try3"); */
    /* ok(j == 2, "Point in cell (node) Y try3"); */

    /* xx = 464375.992; */
    /* yy = 7337128.0; */

    /* ier = sucu_ij_from_xy(&i, &j, &rx, &ry, xx, yy, xori, xinc, yori, yinc, */
    /*     		  mx, my, yflip, rot, 1, debug); */
    /* ok(i == 5, "Point in cell (node) X try3 flag 0"); */
    /* ok(j == 2, "Point in cell (node) Y try3 flag 0"); */

    /* /\* flag = 0 *\/ */
    /* ier = sucu_ij_from_xy(&i, &j, &rx, &ry, xx, yy, xori, xinc, yori, yinc, */
    /* 			  mx, my, yflip, rot, 0, debug); */
    /* ok(i == 6, "Point in cell (node) X try3 flag 1"); */
    /* ok(j == 2, "Point in cell (node) Y try3 flag 1"); */


    /* xx = 464375.609; */
    /* yy = 7337202.060; */

    /* ier = sucu_ij_from_xy(&i, &j, &rx, &ry, xx, yy, xori, xinc, yori, yinc, */
    /* 			  mx, my, yflip, rot, 1, debug); */

    /* ok(i == 3, "Point in cell (node) X try4"); */
    /* ok(j == 3, "Point in cell (node) Y try4"); */

    /* /\* outside point *\/ */
    /* xx = 460000.00; */
    /* yy = 7100000.00; */

    /* ier = sucu_ij_from_xy(&i, &j, &rx, &ry, xx, yy, xori, xinc, yori, yinc, */
    /* 			  mx, my, yflip, rot, 1, debug); */

    /* ok(ier == -1, "Point outside"); */

    /* /\* */
    /*  * ------------------------------------------------------------------------- */
    /*  * Test get point Z from map based on input X Y */
    /*  * ------------------------------------------------------------------------- */
    /*  *\/ */

    /* xx = 464375.992; */
    /* yy = 7337128.076; */

    /* zz = surf_get_z_from_xy(xx, yy, mx, my, xori, yori, xinc, yinc, yflip, rot, */
    /*     		    xxmap, mx*my, debug); */
    /* diff = fabs(zz - 2766.961851); */
    /* ok(diff < 0.0001, "Point sample: surf_get_z_from_xy (1)"); */

    /* xx = 465620.828; */
    /* yy = 7337077.792; */

    /* zz = surf_get_z_from_xy(xx, yy, mx, my, xori, yori, xinc, yinc, yflip, rot, */
    /*     		    xxmap, mx*my, debug); */
    /* diff = fabs(zz - 2516.970); */
    /* ok(diff < 0.0001, "Point sample: surf_get_z_from_xy (2)"); */


    done_testing();

}
