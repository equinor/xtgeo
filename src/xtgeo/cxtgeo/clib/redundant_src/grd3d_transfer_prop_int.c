/*
 * ############################################################################
 * Author: JCR
 * ############################################################################
 * Copy from one integer array to another, but work within a sub range
 * From a larger grid to a reduced one (layer wise)
 * ############################################################################
 */

/*
 *******************************************************************************
 *
 * NAME:
 *    grd3d_transfer_prop_int.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Transfer an INT property from one layer range to another layer range.
 *    It can be layer ranges in two diffeent grid instances. Note that the
 *    range interval must be identical!
 *
 * ARGUMENTS:
 *    nx,ny          i     Grid dimensions I J
 *    nz1            i     NZ input
 *    nz2            i     NZ output
 *    k1min, k1max   i     Range for K in input
 *    k2min, k2max   i     Range for K in output
 *    p_input_v      i     Grid zone parameter input
 *    p_output_v     o     Grid zone parameter to be updated
 *    option         i     Options flag for later usage
 *    debug          i     Debug level
 *
 * RETURNS:
 *    Void + Changed pointer to grid property
 *
 * TODO/ISSUES/BUGS:
 *
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 *******************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"


void grd3d_transfer_prop_int(
			     int   nx,
			     int   ny,
			     int   nz1,
			     int   nz2,
			     int   k1min,
			     int   k1max,
			     int   k2min,
			     int   k2max,
			     int   *p_input_v,
			     int   *p_output_v,
			     int   option,
			     int   debug
			     )

{
    /* locals */
    int i, j, k, ib, ibn, kshift;
    int ierr;
    char s[24]="grd3d_transfer_prop_int";

    ierr = 0;
    if (k1min > k1max || k1min > k1max) ierr=1;
    if (k1min < 1 || k2min < 1 || k1max > nz1 || k2max > nz2 ) ierr=2;

    if ((k2max-k2min) != (k1max-k1min)) ierr=3;

    if (ierr>0) {
	xtg_speak(s,1,"K1MIN/K1MAX K2MIN/K2MAX NZ1 NZ2: %d/%d %d/%d %d %d",
		  k1min, k1max, k2min, k2max, nz1, nz2);

	xtg_error(s,"Routine <%s> reports error code %d", s, ierr);
    }

    kshift=k2min-k1min;

    for (j=1; j<=ny; j++) {
	for (i=1; i<=nx; i++) {
	    for (k=k1min; k<=k1max; k++) {
		ib  = x_ijk2ib(i,j,k,nx,ny,nz1,0);
		ibn = x_ijk2ib(i,j,k+kshift,nx,ny,nz2,0);

		p_output_v[ibn]=p_input_v[ib];
	    }
	}
    }


    xtg_speak(s,2,"Exit transfer integer array");
}
