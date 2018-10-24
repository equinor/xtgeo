/*
 * ############################################################################
 * grd3d_print_cellinfo.c
 * ############################################################################
 * $Id: grd3d_print_cellinfo.c,v 1.1 2001/03/14 08:02:29 bg54276 Exp $
 * $Source: /d/proj/bg/avresmod/src/gplib/GPLExt/RCS/grd3d_print_cellinfo.c,v $
 *
 * $Log: grd3d_print_cellinfo.c,v $
 * Revision 1.1  2001/03/14 08:02:29  bg54276
 * Initial revision
 *
 * Revision 1.1  2000/12/12 17:24:54  bg54276
 * Initial revision
 *
 *
 * ############################################################################
 */


#include "libxtg.h"
#include "libxtg_.h"

/*
 * ############################################################################
 *                         GRD3D_PRINT_CELLINFO
 * ############################################################################
 * Display GRD3D cell info; both geometry and parameters
 * ############################################################################

 */

void grd3d_print_cellinfo (
			   int   i1,
			   int   i2,
			   int   j1,
			   int   j2,
			   int   k1,
			   int   k2,
			   int   nx,
			   int   ny,
			   int   nz,
			   double *p_coord_v,
			   double *p_zcorn_v,
			   int   *actnum_v,
			   int   debug
			   )

{

    /* locals */
    int i, j, k, ic;
    double corners[24];

    xtgverbose(debug);

    for (k = k1; k <= k2; k++) {
	for (j = j1; j <= j2; j++) {
	    for (i = i1; i <= i2; i++) {

		grd3d_corners (
			       i,
			       j,
			       k,
			       nx,
			       ny,
			       nz,
			       p_coord_v,
			       p_zcorn_v,
			       corners,
			       debug
			       );

		printf("==========================================\n");
		printf("Corners for cell %d %d %d\n",i,j,k);
		for (ic=1;ic<=8;ic++) {
		    printf("Corner   %4d:   %13.2f   %13.2f   %7.3f\n",ic,corners[3*ic-3],
			   corners[3*ic-2],corners[3*ic-1]);
		}
	    }
	}
    }
}
