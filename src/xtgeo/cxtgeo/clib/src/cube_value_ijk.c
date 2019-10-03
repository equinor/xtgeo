/*
****************************************************************************************
 *
 * NAME:
 *    cube_value_ijk.c
 *
 * DESCRIPTION:
 *    Given I J K, return cube value. If -UNDEF value is returned. Note here
 *    the cube node is treated as a cell center.
 *
 * ARGUMENTS:
 *    i j k          i     Position in cube
 *    nx ny nz       i     Cube dimensions
 *    p_val_v        i     3D cube values
 *    value          i     Updated value
 *
 * RETURNS:
 *    Function:  0: upon success. If problems:
 *              -1: Some problems with invalid IB
 *    Result value is updated
 *
 * TODO/ISSUES/BUGS:
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 ***************************************************************************************
 */


#include "libxtg.h"
#include "libxtg_.h"

int cube_value_ijk(
                   int   i,
                   int   j,
                   int   k,
                   int   nx,
                   int   ny,
                   int   nz,
                   float *p_val_v,
                   float *value
                   )
{
    /* locals */
    long ib;

    ib = x_ijk2ic(i, j, k, nx, ny, nz, 0);

    if (ib<0) {
	// logger_warn("Problem in routine %s! Outside?", __FUNCTION__);
        *value = UNDEF;
	return(-1);
    }
    else{
	*value = p_val_v[ib];
        return EXIT_SUCCESS;
    }

}
