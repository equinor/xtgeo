/*
 ******************************************************************************
 *
 * NAME:
 *    cube_resample_cube.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *     Given two cubes, 1, and 2, resample values of no 2 into no 1.
 *
 * ARGUMENTS:
 *    ncx1...ncz1      i     cube dimensions for cube 1
 *    cxori1...czinc1  i     cube origin + increment in xyz
 *    crotation1       i     Cube rotation
 *    yflip1           i     Cube YFLIP index
 *    p_cubeval1_v    i/o    1D Array of cube1 values.. to be updated!
 *    ncube1           i     Length of cube2 array
 *    ncx2...ncz2      i     cube dimensions for cube 2
 *    cxori2...czinc2  i     cube origin + increment in xyz
 *    crotation2       i     Cube rotation
 *    yflip2           i     Cube YFLIP index
 *    p_cubeval2_v     i     1D Array of cube2 values of ncx*ncy*ncz size
 *    ncube            i     Length of cube2 array
 *    option1          i     Options:
 *                           0: use cube cell value (no interpolation;
 *                              nearest node)
 *                           1: trilinear interpolation in cube
 *    option2          i     0: Leave cube as-is if outside cube
 *                           1: Set cube values to a spesiric number if outsie
 *    ovalue           i     Value to use if option2 = 1
 *    debug            i     Debug level
 *
 * RETURNS:
 *    Function: 0: upon success. If problems <> 0:
 *          - 4 less than 10% sampled
 *          - 5 No cells sampled
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ******************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"

int cube_resample_cube(
                       int ncx1,
                       int ncy1,
                       int ncz1,
                       double cxori1,
                       double cxinc1,
                       double cyori1,
                       double cyinc1,
                       double czori1,
                       double czinc1,
                       double crotation1,
                       int yflip1,
                       float *p_cubeval1_v,
                       long ncube,
                       int ncx2,
                       int ncy2,
                       int ncz2,
                       double cxori2,
                       double cxinc2,
                       double cyori2,
                       double cyinc2,
                       double czori2,
                       double czinc2,
                       double crotation2,
                       int yflip2,
                       float *p_cubeval2_v,
                       long ncube2,
                       int option1,
                       int option2,
                       float ovalue,
                       int debug
                       )

{
    /* locals */
    char s[24] = "cube_resample_cube";
    int ic1, jc1, kc1;
    int ier;
    long icn1, nm = 0;
    double xc, yc, zc;
    float value;


    xtgverbose(debug);

    /* work with every cube1 node */
    for (ic1 = 1; ic1 <= ncx1; ic1++) {
        if (debug > 2 ) xtg_speak(s, 3, "Working with cube IL %d of %d ...",
                                  ic1, ncx1);
        for (jc1 = 1; jc1 <= ncy1; jc1++) {
            for (kc1 = 1; kc1 <= ncz1; kc1++) {

                /* get the cube x, y, z for i j */
                ier = cube_xy_from_ij(ic1, jc1, &xc, &yc, cxori1, cxinc1,
                                      cyori1, cyinc1, ncx1, ncy1, yflip1,
                                      crotation1, 0, debug);

                zc = czori1 + czinc1 * (kc1 - 1);

                icn1 = x_ijk2ic(ic1, jc1, kc1, ncx1, ncy1, ncz1, 0);

                if (option1 == 0) {

                    ier = cube_value_xyz_cell(xc, yc, zc, cxori2, cxinc2,
                                              cyori2,
                                              cyinc2, czori2, czinc2,
                                              crotation2,
                                              yflip2, ncx2, ncy2, ncz2,
                                              p_cubeval2_v, &value, 0,
                                              debug);
                }
                else if (option1 == 1) {

                    ier = cube_value_xyz_interp(xc, yc, zc, cxori2, cxinc2,
                                                cyori2,
                                                cyinc2, czori2, czinc2,
                                                crotation2,
                                                yflip2, ncx2, ncy2, ncz2,
                                                p_cubeval2_v, &value, 0,
                                                debug);


                }
                else{
                    xtg_error(s, "Invalid option1 (%d) to %s", option1, s);
                }


                if (ier == EXIT_SUCCESS) {
                    p_cubeval1_v[icn1] = value;
                    nm++;
                }
                else if (ier == -1 && option2 == 0) {
                    /* option2 = 0 shall just keep cube value as is */
                    if (debug > 3) xtg_speak(s, 4, "Keep value as is");
                }
                else if (ier == -1 && option2 == 1) {
                    /* option2 = 1 Use another value */
                    p_cubeval1_v[icn1] = ovalue;
                }
            }
        }
    }
    /* less than 10% sampled */
    if (nm > 0 && nm < 0.1*ncube2) {
        xtg_warn(s, 1, "Less than 10\% nodes sampled in %s!", s);
        return -4;
    }

    /* no nodes sampled */
    if (nm == 0) {
        xtg_warn(s, 1, "No nodes sampled in %s!", s);
        return -5;
    }

    return EXIT_SUCCESS;
}
