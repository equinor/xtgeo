/*
 * ############################################################################
 * grd3d_collapse_inact.c
 * ############################################################################
 */


#include "libxtg.h"
#include "libxtg_.h"

/*
 * ############################################################################
 * Look at inactive cells, and collapse the corners to a mid value
 * ############################################################################
 */

void grd3d_collapse_inact (
			   int   nx,
			   int   ny,
			   int   nz,
			   double *p_zcorn_v,
			   int   *p_actnum_v,
			   int   debug
			   )

{
    /* locals */
    int     i, j, k, ic, ibp, ibx, iflag, kk, kkk, k2=0;
    double  z1, z2;
    char    sub[24]="grd3d_collapse_inact";

    xtgverbose(debug);

    xtg_speak(sub,2,"Entering <grd3d_collapse_inact>");
    
    for (j = 1; j <= ny; j++) {
	xtg_speak(sub,3,"Finished column %d of %d",j,ny);

	for (i = 1; i <= nx; i++) {
	    iflag=0;
	    /* check that column has active cells */
	    for (k = 2; k <= nz+1; k++) {


		ibp=x_ijk2ib(i,j,k-1,nx,ny,nz+1,0);
		if (p_actnum_v[ibp]==1) { 
		    iflag=1;
		}


	    }

	    if (iflag==1) {

		for (k = 2; k <= nz+1; k++) {
		    ibp=x_ijk2ib(i,j,k-1,nx,ny,nz+1,0);
		    /* find inactive cell */

		    if (p_actnum_v[ibp]==0) {

			/* find next active cell */
			for (kk = k; kk <= nz+1; kk++) {	
		
			    if (kk < nz+1) {
				ibx=x_ijk2ib(i,j,kk,nx,ny,nz+1,0);
				
				if (p_actnum_v[ibx]==1) {
				    k2=kk;
				    break;
				}
			    }
			}
			/* check each corner */

			ibx=x_ijk2ib(i,j,k2,nx,ny,nz+1,0);
			for (ic=1;ic<=4;ic++) {
			    z1=p_zcorn_v[4*ibp + 1*ic - 1];
			    z2=p_zcorn_v[4*ibx + 1*ic - 1];
			    if ((z2-z1) > 0.0) { 
				/* k-1 */
				p_zcorn_v[4*ibp + 1*ic - 1] = 0.5*(z1 + z2);
				/* all the other below */
				for (kkk= k; kkk <= k2; kkk++) {
				    ibx=x_ijk2ib(i,j,kkk,nx,ny,nz+1,0);
				    p_zcorn_v[4*ibx + 1*ic - 1] = 0.5*(z1 + z2);
				}			       
			    }
			}
		    }
		}
	    }
	}
    }
    xtg_speak(sub,2,"Exiting <grd3d_collapse_inact>");
}



