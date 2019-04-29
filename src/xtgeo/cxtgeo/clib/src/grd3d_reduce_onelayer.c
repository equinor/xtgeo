/*
 *******************************************************************************
 *
 * NAME:
 *    grd3d_reduce_onelayer.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Reduce the grid to one single big layer
 *
 * ARGUMENTS:
 *    nx,ny,nz       i     Grid dimensions I J K in input
 *    p_zcorn1_v     i     Grid Z corners for input
 *    p_zcorn2_v     o     Grid Z corners for output
 *    p_actnum1_v    i     Grid ACTNUM parameter input
 *    p_actnum2_v    o     Grid ACTNUM parameter output
 *    nactive        o     Number of active cells
 *    iflag          i     Options flag (future use)
 *    debug          i     Debug level
 *
 * RETURNS:
 *    The C macro EXIT_SUCCESS unless problems + changed pointers
 *
 * TODO/ISSUES/BUGS:
 *    ACTNUM is set to 1 for all cells (iflag=0), only.
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 *******************************************************************************
 */


#include "libxtg.h"
#include "libxtg_.h"


int grd3d_reduce_onelayer (
			   int    nx,
			   int    ny,
			   int    nz,
			   double *p_zcorn1_v,
			   double *p_zcorn2_v,
			   int    *p_actnum1_v,
			   int    *p_actnum2_v,
			   int    *nactive,
			   int    iflag,
			   int    debug
			   )
{
    /* locals */
    char s[24]="grd3d_reduce_onelayer";
    int  i, j, ic, ib, ibt, ibb, ncc;

    xtgverbose(debug);

    xtg_speak(s,1,"Entering routine <%s>",s);

    xtg_speak(s,2,"Map Z corners, top and base...");

    for (j = 1; j <= ny; j++) {
	for (i = 1; i <= nx; i++) {
	    /* top */
	    ibt=x_ijk2ib(i,j,1,nx,ny,nz+1,0);
	    ibb=x_ijk2ib(i,j,1,nx,ny,2,0);

	    for (ic=1;ic<=4;ic++) {
		p_zcorn2_v[4*ibb + 1*ic - 1] = p_zcorn1_v[4*ibt + 1*ic - 1];
	    }

	    /* base */
	    ibt=x_ijk2ib(i,j,nz+1,nx,ny,nz+1,0);
	    ibb=x_ijk2ib(i,j,2,nx,ny,2,0);

	    for (ic=1;ic<=4;ic++) {
		p_zcorn2_v[4*ibb + 1*ic - 1] = p_zcorn1_v[4*ibt + 1*ic - 1];
	    }
	}
    }


    /* transfer actnum */
    ncc=0;

    if (iflag==0) {
	xtg_speak(s,2,"ACTNUM = 1 for all cells...");
	for (ib=0; ib<nx*ny*1; ib++) {
	    p_actnum2_v[ib]=1;
	    ncc++;
	}
    }
    else{
	xtg_error(s,"IFLAG other than 0 not implemented yet for <%s>",s);
    }

    *nactive=ncc;


    xtg_speak(s,1,"Exit from <%s>",s);

    return EXIT_SUCCESS;

}
