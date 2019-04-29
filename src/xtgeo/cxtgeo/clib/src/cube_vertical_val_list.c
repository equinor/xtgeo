/*
 ******************************************************************************
 *
 * Get a list of vertical cube values given I J
 *
 ******************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"

/*
 ******************************************************************************
 *
 * NAME:
 *    cube_vertical_list.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Given I J, sample the verticla values (trace) in a cube The array
 *    will start from top, and start with index 0 (as all 1D arrays should do)
 *
 * ARGUMENTS:
 *    i j            i     Input location
 *    nx, ny, nz     i     Cube dimensions
 *    p_val_v        i     Cube values (3D)
 *    p_vertical_v   o     The trace
 *    debug          i     Debug level
 *
 * RETURNS:
 *    Function:  0: upon success.
 *              -1: IB counter out of range (fatal)
 *    Result p_vertical_v is updated
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 *******************************************************************************
 */
int cube_vertical_val_list(
			   int   i,
			   int   j,
			   int   nx,
			   int   ny,
			   int   nz,
			   float *p_val_v,
			   float *p_vertical_v,
			   int   debug
			   )
{
    /* locals */
    char     sub[24]="cube_vertical_val_list";
    long     k, ib;


    xtgverbose(debug);
    xtg_speak(sub,2,"Entering routine");

    for (k=1; k<=nz; k++) {
	ib=x_ijk2ic(i, j, k, nx, ny, nz, 0);

	if (ib<0) {
	    xtg_warn(sub,1,"Problem in routine %s!",&sub);
	    return -1;
	}
	else{
	    p_vertical_v[k-1]=p_val_v[ib];
	}

    }
    return 0;

}
