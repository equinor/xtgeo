/*
 * ############################################################################
 * grd3d_set_dz_cell.c
 * ############################################################################
 * $Id: $ 
 * $Source: $ 
 *
 * $Log: $
 *
 * ############################################################################
 */


#include "libxtg.h"
#include "libxtg_.h"

/*
 * ############################################################################
 * Set DZ to a specific valye for a given cell. At the moment, dont care that
 * surrouning cells will be not lined up; e.g. we introduce a "fault"
 * ############################################################################
 */

void grd3d_set_dz_cell (
			int   i,
			int   j,
			int   k,
			int   nx,
			int   ny,
			int   nz,
			double *p_zcorn_v,
			int   *p_actnum_v,
			double zsep,
			int   debug
			)

{
    /* locals */
    int    ic, ibp, ibx;
    double  z1;
    char sub[24]="grd3d_set_dz_cell";

    xtgverbose(debug);

    xtg_speak(sub,2,"Entering routine");
    xtg_speak(sub,4,"Minimum cell Z seperation is %f", zsep);
    
    ibp=x_ijk2ib(i,j,k,nx,ny,nz+1,0);
    ibx=x_ijk2ib(i,j,k+1,nx,ny,nz+1,0);

    /* each corner */
    for (ic=1;ic<=4;ic++) {
      z1=p_zcorn_v[4*ibp + 1*ic - 1];
      p_zcorn_v[4*ibx + 1*ic - 1] = z1 + zsep;
      /* TODO fix neighbours */
    }
    
    xtg_speak(sub,2,"Exiting routine");
}



