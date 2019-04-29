/*
 *******************************************************************************
 *
 * NAME:
 *    grd3d_reduce_by_zon.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Reduce the grid by using a zonation interval
 *
 * ARGUMENTS:
 *    nx,ny          i     Grid dimensions I J
 *    nz1            i     NZ as input
 *    nz2            o     NZ for output
 *    p_zcorn1_v     i     Grid Z corners for input
 *    p_zcorn2_v     o     Grid Z corners for output
 *    p_actnum1_v    i     Grid ACTNUM parameter input
 *    p_actnum2_v    o     Grid ACTNUM parameter output
 *    p_zon1_v       i     Grid zone parameter input
 *    p_zon2_v       o     Grid zone parameter input
 *    iflag          i     Options flag
 *    debug          i     Debug level
 *
 * RETURNS:
 *    The C macro EXIT_SUCCESS unless problems + changed pointers
 *
 * TODO/ISSUES/BUGS:
 *    Code is not finished
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 *******************************************************************************
 */


#include "libxtg.h"
#include "libxtg_.h"


int grd3d_reduce_by_zon (
			 int    zmin,
			 int    zmax,
			 int    nx,
			 int    ny,
			 int    nz1,
			 int    *nz2,
			 double *p_zcorn1_v,
			 double *p_zcorn2_v,
			 int    *p_actnum1_v,
			 int    *p_actnum2_v,
			 int    *p_zon1_v,
			 int    *p_zon2_v,
			 int    *nactive,
			 int    *kminimum,
			 int    *kmaximum,
			 int    iflag,
			 int    debug
			 )
{
    /* locals */
    char s[24]="grd3d_reduce_by_zon";
    int  ib, ib1, ib2, ic, i, j, k, ibt, ibb, ibp;
    int  kmin, kmax, knew, ncc, nnz2;

    xtgverbose(debug);

    ib1 = nx*ny*nz1+99;
    ib2 = 0;

    kmin=nz1;
    kmax=0;

    xtg_speak(s,1,"Entering routine <%s>",s);

    xtg_speak(s,2,"Find geometric numbers...");

    /* first loop for geometric numbers (kmin, kmax) */
    for (k = 1; k <= nz1; k++) {
	xtg_speak(s,3,"Find geometric numbers...K = %d",k);
	for (j = 1; j <= ny; j++) {
	    for (i = 1; i <= nx; i++) {
		ibp=x_ijk2ib(i,j,k,nx,ny,nz1,0);

		if (p_actnum1_v[ibp] == 1 && p_zon1_v[ibp] >= zmin &&
		    p_zon1_v[ibp] <= zmax) {

		    if (kmin > k) kmin = k;
		    if (kmax < k) kmax = k;

		}
	    }
	}
    }

    xtg_speak(s,2,"Find geometric numbers... done");

    *nz2 = kmax - kmin + 1;
    *kminimum = kmin;
    *kmaximum = kmax;

    /* a dry run first is to find K range ie the nz2 pointer*/
    if (iflag == 1){
	xtg_speak(s,1,"Exit from <%s> with iflag=1",s);
	return EXIT_SUCCESS;
    }


    xtg_speak(s,2,"Map Z corners...");

    /* do mappng over the needed ZCORNS */
    for (k = kmin; k <= kmax+1; k++) {
	knew = k - kmin + 1;
	for (j = 1; j <= ny; j++) {
	    for (i = 1; i <= nx; i++) {
		ibt=x_ijk2ib(i,j,k,nx,ny,nz1+1,0);
		ibb=x_ijk2ib(i,j,knew,nx,ny,*nz2+1,0);

		/* do for all corners */
		for (ic=1;ic<=4;ic++) {
		   p_zcorn2_v[4*ibb + 1*ic - 1] = p_zcorn1_v[4*ibt + 1*ic - 1];
		}
	    }
	}
    }

    xtg_speak(s,2,"New ACTNUM... range %d  %d", kmin, kmax);

    nnz2 = *nz2;

    /* transfer actnum */
    grd3d_transfer_prop_int(nx, ny, nz1, nnz2, kmin, kmax, 1, nnz2,
			    p_actnum1_v, p_actnum2_v, 0, debug);

    ncc=0;
    for (ib=0; ib<nx*ny*nnz2; ib++) {
	if (p_actnum2_v[ib]==1) ncc++;
    }

    *nactive=ncc;


    xtg_speak(s,1,"Exit from <%s>",s);

    return EXIT_SUCCESS;

}
