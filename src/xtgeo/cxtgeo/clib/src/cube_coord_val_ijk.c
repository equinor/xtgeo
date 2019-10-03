/*
****************************************************************************************
 *
 * NAME:
 *    cube_coord_ijk.c
 *
 * DESCRIPTION:
 *    Given I J K, return cube coordinates and value.
 *
 * ARGUMENTS:
 *    i j k          i     Position in cube
 *    nx ny nz       i     Cube dimensions
 *    p_val_v        i     3D cube values
 *    xori...zinc    i     Cube coordinates
 *    rot_deg        i     Cube rotation
 *    yflip          i     yflip flag
 *    p_val_v        i     value array
 *    xcor .. zcor   o     Cube coordinates in cell IJK
 *    value          o     Cube value in IJK
 *    option         i     If option >= 10, then x y calc is skipped
 *
 * RETURNS:
 *    Function:  0: upon success. If problems:
 *              -1: Some problems...
 *    Result x y z value is updated
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ***************************************************************************************
 */


#include "libxtg.h"
#include "libxtg_.h"

int cube_coord_val_ijk(
                       int   i,
                       int   j,
                       int   k,
                       int   nx,
                       int   ny,
                       int   nz,
                       double xori,
                       double xinc,
                       double yori,
                       double yinc,
                       double zori,
                       double zinc,
                       double rot_deg,
                       int yflip,
                       float *p_val_v,
                       double *xcor,
                       double *ycor,
                       double *zcor,
                       float *value,
                       int option
                       )
{

    /* locals */
    int ier1, ier2;
    static double xcoord = 0.0, ycoord = 0.0;

    /* find coordinates: */

    ier1 = 0;
    if (option < 10) {
        ier1 = cube_xy_from_ij(i, j, &xcoord, &ycoord, xori, xinc, yori,
                               yinc, nx, ny, yflip, rot_deg, 0);
    }
    *xcor = xcoord;
    *ycor = ycoord;

    if (ier1 != 0) exit(-1);

    *zcor = zori + (k - 1) * zinc;

    /* and now update the value: */
    ier2 = cube_value_ijk(i, j, k, nx, ny, nz, p_val_v, value);

    if (ier2 == -1 && ier1 == 0) {
        return ier2;
    }

    if (ier1 == 0 && ier2 == 0) {
        return EXIT_SUCCESS;
    }
    else{
        /* something is wrong */
        printf("IER1 = %d IER2 = %d Error(?) in routine"
               " %s contact JRIV", ier1, ier2, __FUNCTION__);
        *value = UNDEF;
        return -1;
    }

}
