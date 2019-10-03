
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
		    int flag
		    )
{
    /* locals */
    int ier=0;
    double   p_dummy, zdum;

    /* reuse routine; set flag = 1 so theta p_map_v is not applied */
    ier = surf_xyz_from_ij(i, j, x, y, &zdum, xori, xinc, yori, yinc,
                           nx ,ny, yflip, rot_deg, &p_dummy, 1, 1);

    if (ier != 0) return ier;

    return EXIT_SUCCESS;
}
