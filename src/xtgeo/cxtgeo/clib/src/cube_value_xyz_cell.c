/*
 ******************************************************************************
 *
 * NAME:
 *    cube_value_xyz_cell.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Given X Y Z, return cell (nearest point) cube value.
 *
 * ARGUMENTS:
 *    x, y, z        i     Position in cube to request a value
 *    xinc..rot_deg  i     Cube geometry description
 *    yflip          i     If the cube is flipped in Y (1 or -1)
 *    nx ny nz       i     Cube dimensions
 *    p_val_v        i     3D cube values
 *    value          o     Updated cube cell value
 *    option         i     For later use
 *    debug          i     Debug level
 *
 * RETURNS:
 *    Function:  0: upon success. If problems:
 *              -1: Some problems with invalid IB (outside cube)
 *    Result value (pointer) is updated if success
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ******************************************************************************
 */


#include "libxtg.h"
#include "libxtg_.h"

int cube_value_xyz_cell(
                        double x,
                        double y,
                        double z,
                        double xori,
                        double xinc,
                        double yori,
                        double yinc,
                        double zori,
                        double zinc,
                        double rot_deg,
                        int yflip,
                        int nx,
                        int ny,
                        int nz,
                        float *p_val_v,
                        float *value,
                        int option,
                        int debug
                        )
{
    /* locals */
    char s[24]="cube_value_xyz_cell";
    int  i, j, k, ier, istat;
    float val;
    double rx, ry, rz;


    xtgverbose(debug);

    if (debug > 2) xtg_speak(s,3,"Entering routine %s", s);

    /* first get IJK value from XYZ point */
    istat = cube_ijk_from_xyz(&i, &j, &k, &rx, &ry, &rz, x, y, z, xori, xinc,
                              yori, yinc, zori, zinc,
                              nx, ny, nz, rot_deg, yflip, 0,
                              debug
                              );

    /* now get the cube cell value in IJK */
    if (istat == 0) {
        ier = cube_value_ijk(i, j, k, nx, ny, nz,
                             p_val_v, &val, debug);
        *value = val;

        if (debug > 2) {
            xtg_speak(s,3,"Cube I J K and value is %d %d %d ==> %f",
                      i, j, k, val);
        }
    }
    else{
        *value = UNDEF;
        return -1;
    }


    return EXIT_SUCCESS;
}
