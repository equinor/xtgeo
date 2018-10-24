/*
 * ############################################################################
 * grd3d_calc_cell_dip.c
 * Calculates the maximum dip a cell has
 * Author: JCR
 * ############################################################################
 * $Id: $
 * $Source: $
 *
 * $Log: $
 *
 *
 * ############################################################################
 */


#include <math.h>
#include "libxtg.h"
#include "libxtg_.h"


void grd3d_calc_cell_dip(
			 int nx,
			 int ny,
			 int nz,
			 double *p_coord_v,
			 double *p_zcorn_v,
			 double *p_dip_v,
			 int   debug
			 )

{
    /* locals */

    int     i, j, k, ib, ic, ip, iq, kmin=1, kmax=99999;
    double  xlen,ylen,xylen, zmin, zmax, angle;
    double  zavg[5], corner_v[24];
    char    s[24]="grd3d_calc_cell_dip";

    xtgverbose(debug);
    xtg_speak(s,2,"Entering <grd3d_calc_cell_dip>");
    xtg_speak(s,3,"NX NY NZ: %d %d %d", nx, ny, nz);


    xtg_speak(s,2,"Finding grid dip parameter...");

    for (k = 1; k <= nz; k++) {
	xtg_speak(s,3,"Finished layer %d of %d",k,nz);
	for (j = 1; j <= ny; j++) {
	    for (i = 1; i <= nx; i++) {

		/* parameter counting */
		ib=x_ijk2ib(i,j,k,nx,ny,nz,0);

		/* grid */
		ip=x_ijk2ib(i,j,k,  nx,ny,nz+1,0);
		iq=x_ijk2ib(i,j,k+1,nx,ny,nz+1,0);

		/* each cell: find avg Z for each corner */
		zavg[1] = 0.5*(p_zcorn_v[4*ip + 1 - 1]+
			       p_zcorn_v[4*iq + 1 - 1]);
		zavg[2] = 0.5*(p_zcorn_v[4*ip + 2 - 1]+
			       p_zcorn_v[4*iq + 2 - 1]);
		zavg[3] = 0.5*(p_zcorn_v[4*ip + 3 - 1]+
			       p_zcorn_v[4*iq + 3 - 1]);
		zavg[4] = 0.5*(p_zcorn_v[4*ip + 4 - 1]+
			       p_zcorn_v[4*iq + 4 - 1]);

		/*find min and max values */
		zmin=9999999.00;
		zmax=-9999999.00;
		for (ic=1;ic<=4;ic++) {
		    if (zavg[ic]<zmin) {
			zmin=zavg[ic];
			kmin=ic;
		    }
		    if (zavg[ic]>zmax) {
			zmax=zavg[ic];
			kmax=ic;
		    }
		}

		/* now I need angles, and delta(DIST), which depends...*/
		grd3d_corners(i,j,k,nx,ny,nz,p_coord_v,
			      p_zcorn_v,corner_v,debug);

		xlen=fabs(0.5*(corner_v[kmin*3-3]+corner_v[kmin*3+9]) -
			  0.5*(corner_v[kmax*3-3]+corner_v[kmax*3+9]));

		ylen=fabs(0.5*(corner_v[kmin*3-2]+corner_v[kmin*3+10]) -
			  0.5*(corner_v[kmax*3-2]+corner_v[kmax*3+10]));

		xylen=sqrt(pow(xlen,2)+pow(ylen,2));

		if (xylen > 0.0001) {
		    angle=(180/3.1415926)*atan((zmax-zmin)/xylen);
		}
		else{
		    angle=0.0;
		}

		p_dip_v[ib]=angle;
	    }
	}
    }
    xtg_speak(s,2,"Exiting <grd3d_calc_cell_dip>");
}
