/*
 * ############################################################################
 * cube_xy_from_ij.c
 *
 * Routine(s) computing XY in a rotated cube given I,J. Note the coordinate
 * is a node value in Cube context , but in a grid context it is a mid point.
 *
 * ############################################################################
 * ToDo:
 * -
 * ############################################################################
 * ############################################################################
 */

#include "libxtg.h"
#include "libxtg_.h"

int cube_xy_from_ij(
		    int i,
		    int j,
		    double *x,
		    double *y,
		    double xori,
		    double xinc,
		    double yori,
		    double yinc,
		    int nx,
		    int ny,
                    int yflip,
		    double rot_deg,
		    int flag,
		    int debug
		    )
{
    /* locals */
    char     s[24]="cube_xy_from_ij";
    int ier=0;
    double   p_dummy, zdum;

    xtgverbose(debug);
    if (debug > 2) xtg_speak(s, 3, "Entering routine %s", s);

    if (debug > 2) {
        xtg_speak(s,3,"Input: I, J, YFLIP is %d %d %d", i, j, yflip);
        xtg_speak(s,3,"Input: NX NY, rotation %d %d %f", nx, ny, rot_deg);
        xtg_speak(s,3,"XORI XINC YORI YINC  %f %f %f %f",
                  xori, xinc, yori, yinc);
    }

    /* reuse routine; set flag = 1 so theta p_map_v is not applied */
    ier = surf_xyz_from_ij(i, j, x, y, &zdum, xori, xinc, yori, yinc,
                           nx ,ny, yflip, rot_deg, &p_dummy, 1, 1, debug);

    if (ier != 0) return ier;

    return EXIT_SUCCESS;
}
