/*
 ******************************************************************************
 *
 * NAME:
 *    grd3d_corners.c
 *
 * DESCRIPTION:
 *    Given a cell coordinate I J K, find all corner coordinates as an
 *    array with 24 values
 *
 *      Top  --> i-dir     Base cell
 *
 *  6,7,8   9,10,11  18,19,20   21,22,23      0 = X, 1 = Y, 2 = Z, etc
 *    |-------|          |-------|
 *    |       |          |       |
 *    |       |          |       |
 *    |-------|          |-------|
 *  0,1,2   3,4,5    12,13,14,  15,16,17
 *
 *
 * ARGUMENTS:
 *    i, j, k        i     Cell number
 *    nx,ny,nz       i     Grid dimensions
 *    p_coord_v      i     Grid Z coord for input
 *    p_zcorn_v      i     Grid Z corners for input
 *    corners        o     Array, 24 length
 *    debug          i     Debug level
 *
 * RETURNS:
 *    Corners
 *
 * TODO/ISSUES/BUGS:
 *    None known
 *
 * LICENCE:
 *    Equinor property
 ******************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"
#include <math.h>

void grd3d_corners (
		    int     i,
		    int     j,
		    int     k,
		    int     nx,
		    int     ny,
		    int     nz,
		    double  *p_coord_v,
		    double  *p_zcorn_v,
		    double  corners[],
		    int     debug
		    )

{
    int    ic, cl, im, jm, ibb, ibt;
    double xtop[5], ytop[5], ztop[5];
    double xbot[5], ybot[5], zbot[5];
    char   s[24] = "grd3d_corners";

    xtgverbose(debug);

    /* each cell is defined by 4 pillars */

    for (ic = 1; ic <= 4; ic++) {
	jm = 0;
	im = 0;
	if (ic == 1 || ic == 2) jm = 1;
	if (ic == 1 || ic == 3) im = 1;

	xtop[ic] = p_coord_v[6 * ((j - jm) * (nx + 1) + i - im) + 0];
	ytop[ic] = p_coord_v[6 * ((j - jm) * (nx + 1) + i - im) + 1];
	ztop[ic] = p_coord_v[6 * ((j - jm) * (nx + 1) + i - im) + 2];
	xbot[ic] = p_coord_v[6 * ((j - jm) * (nx + 1) + i - im) + 3];
	ybot[ic] = p_coord_v[6 * ((j - jm) * (nx + 1) + i - im) + 4];
	zbot[ic] = p_coord_v[6 * ((j - jm) * (nx + 1) + i - im) + 5];

    }

    /* cell and cell below*/
    ibt = x_ijk2ib(i,j,k,nx,ny,nz+1,0);
    ibb = x_ijk2ib(i,j,k+1,nx,ny,nz+1,0);


    corners[2]  = p_zcorn_v[4*ibt + 1*1 - 1];
    corners[5]  = p_zcorn_v[4*ibt + 1*2 - 1];
    corners[8]  = p_zcorn_v[4*ibt + 1*3 - 1];
    corners[11] = p_zcorn_v[4*ibt + 1*4 - 1];

    corners[14] = p_zcorn_v[4*ibb + 1*1 - 1];
    corners[17] = p_zcorn_v[4*ibb + 1*2 - 1];
    corners[20] = p_zcorn_v[4*ibb + 1*3 - 1];
    corners[23] = p_zcorn_v[4*ibb + 1*4 - 1];

    for (ic = 1; ic <= 8; ic++) {
	cl = ic;
	if (ic == 5) cl = 1;
	if (ic == 6) cl = 2;
	if (ic == 7) cl = 3;
	if (ic == 8) cl = 4;

	if (fabs(zbot[cl]-ztop[cl]) > 0.01) {
	    corners[3*(ic-1)+0]=xtop[cl]-(corners[3*(ic-1)+2]-ztop[cl])*
		(xtop[cl]-xbot[cl])/(zbot[cl]-ztop[cl]);
	    corners[3*(ic-1)+1]=ytop[cl]-(corners[3*(ic-1)+2]-ztop[cl])*
		(ytop[cl]-ybot[cl])/(zbot[cl]-ztop[cl]);
	}
	//else if ((zbot[cl]-ztop[cl]) < -0.01) {
	//    xtg_warn(s,1,"Cell I,J,K: %d %d %d", i, j, k);
	//    xtg_warn(s,1,"ZBOT: %f  ZTOP: %f  Cornerline: %d",zbot[cl],ztop[cl], cl);
	//    xtg_error(s,"Pillar is inverted. STOP!");
	//}
	else{
	    corners[3*(ic-1)+0]=xtop[cl];
	    corners[3*(ic-1)+1]=ytop[cl];
	}
    }

    if (debug>3) {
	ibb=0;
	for (ic=0;ic<8;ic++) {
	    xtg_speak(s,4,"Corner %d: ",ic);
	    for (im=0;im<3;im++) {
		xtg_speak(s,4,"Corner coord no %d: %11.2f",im,corners[ibb++]);
	    }
	}
    }


    xtg_speak(s,4,"==== Exiting grd3d_corners ====");

}
