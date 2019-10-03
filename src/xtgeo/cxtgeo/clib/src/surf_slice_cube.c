/*
 ******************************************************************************
 *
 * NAME:
 *    surf_slice_cube.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *     Given a map and a cube, sample cube values to the map and return a
 *     map copy with cube values sampled.
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
 *    p_zslice_v     i     map array with Z values
 *    nslice         i     Length of slice array
 *    p_map_v        o     map to update
 *    nmap           i     Length of map array
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
 *    Function: 0: upon success. If problems <> 0:
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
 * OVERVIEW:
 *
 *    # check for every map node i J, get X Y per node
 *      * if the value is not UNDEF:
 *
 *        @ if nearest node:
 *          <cube_value_xyz_cell>, input X Y Z, return updated value from cube
 *            <cube_ijk_from_xyz>, find cube IJK from map XYZ
 *               <sucu_ij_from_xy>, find IJ from XY, nearest cell mode.
 *                                  and also provide relative coordinates
 *                  <x_point_line_pos>. finds if point is inside cube XY
 *                                      both for X and Y;
 *            <cube_value_ijk>, find cube value from cube IJK
 *            return this value!
 *
 *        @ if trilinar node:
 *          <cube_value_xyz_interp>, input X Y Z, return updated value from cube
 *            <cube_ijk_from_xyz>, find cube IJK from map XYZ
 *               <sucu_ij_from_xy>, find IJ from XY, nearest cell mode.
 *                                  and also provide relative coordinates
 *                  <x_point_line_pos>. finds if point is inside cube XY
 *                                      both for X and Y;
 *            <cube_value_ijk>, find cube value from cube IJK
 *            return this value!
 *
 ******************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"
#include <math.h>

int surf_slice_cube(
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
                    double *p_zslice_v,
                    long nslice,
                    double *p_map_v,
                    long nmap,
                    int option1,
                    int option2,
                    int debug
                    )

{
    /* locals */
    char s[24] = "surf_slice_cube";
    int im, jm, ier;
    long ibm = 0;
    double x, y, z;
    float value;
    int nm = 0, option1a = 0;

    xtgverbose(debug);
    xtg_speak(s, 2, "Entering routine %s", s);


    if (nmap != nslice) {
        xtg_error(s, "Something is plain wrong in %s (nmap vs nslice)", s);
    }

    xtg_speak(s, 2, "Mapflip is %d", mapflip);

    /* work with every map node */
    for (im = 1; im <= mx; im++) {
        if (debug > 2) xtg_speak(s, 3, "Working with map column %d of %d ...",
                                 im, mx);

        for (jm = 1; jm <= my; jm++) {
            if (debug > 2) xtg_speak(s, 3, "... map row %d of %d", jm, my);

            /* get the surface x, y, value (z) from IJ location */
            ier = surf_xyz_from_ij(im, jm, &x, &y, &z, xori, xinc,
                                   yori, yinc, mx, my, mapflip,
                                   mrotation, p_zslice_v, nslice, 0);

            ier = 99;

            ibm = x_ijk2ic(im, jm, 1, mx, my, 1, 0);

            if (z < UNDEF_MAP_LIMIT) {

                if (option1 == 0) {

                    ier = cube_value_xyz_cell(x, y, z, cxori, cxinc, cyori,
                                              cyinc, czori, czinc, crotation,
                                              yflip, ncx, ncy, ncz,
                                              p_cubeval_v, &value, 0);
                }
                else if (option1 == 1 || option1 == 2) {

                    option1a = 0;
                    if (option1 == 2) option1a = 1;  // snap to closest XY

                    ier = cube_value_xyz_interp(x, y, z, cxori, cxinc, cyori,
                                                cyinc, czori, czinc, crotation,
                                                yflip, ncx, ncy, ncz,
                                                p_cubeval_v, &value, option1a);


                }
                else{
                    xtg_error(s, "Invalid option1 (%d) to %s", option1, s);
                }


                if (ier == EXIT_SUCCESS) {
                    p_map_v[ibm] = value;
                    nm ++;
                }
                else if (ier == -1 && option2 == 0) {
                    /* option2 = 1 shall just keep map value as is */
                    p_map_v[ibm] = UNDEF_MAP;
                }
            }
            else{
                p_map_v[ibm] = UNDEF_MAP;
            }
        }
    }

    /* less than 10% sampled */
    if (nm > 0 && nm < 0.1*nmap) {
        xtg_warn(s, 1, "Less than 10%% nodes sampled in %s!", s);
        return -4;
    }

    /* no nodes */
    if (nm == 0) {
        xtg_warn(s, 1, "No nodes sampled in %s!", s);
        return -5;
    }

    return EXIT_SUCCESS;
}
