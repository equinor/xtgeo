/*
 * #############################################################################
 * Name:      grd3d_point_in_cell.c
 * Author:    JRIV@statoil.com
 * Created:   2002?
 * Updates:   2015-09-11 Polished and also expanded to see in 2D only
 * #############################################################################
 * Find which cell (ib) that contains the current point. An input ib gives 
 * a much faster search if the next point is close to the first one
 *
 * Arguments:
 *     ibstart          which IB to start search from
 *     kzonly           A number in [1..nz] if only looking within a 
 *                      cell layer, 0 otherwise
 *     x,y,z            input points. If z is -999 it means a 2D search only
 *     nx..nz           grid dimensions
 *     p_xp_v p_yp_v    polygon array (X Y)
 *     p_coord_v        grid coords
 *     p_zcorn_v        grid zcorn
 *     p_actnum_v       grid active cell indicator
 *     p_prop_v         property to work with
 *     value            value to set
 *     ronly            replace-only-this value
 *     i1, i2...k2      cell index range in I J K
 *     option           0 for looking in cell 3D, 1 for looking in 2D bird view
 *     debug            debug/verbose flag
 *
 * Return:
 *     A value ranging from 0 to nx*ny*nz-1 of found. -1 otherwise

 * Caveeats/issues:
 *     - consistency if ibstart > 0 but not in correct kzonly layer?
 *     - how to use ACTNUM (not used so far)
 * #############################################################################
 */


/*
 * ----------------------------------------------------------------------------
 *
 * CELL CORNERS is a 24 vector long (x y z x y z ....)
 *     ---> I
 *
 *   0 1 2  ------------------ 3 4 5     12 13 14  ------------------ 15 16  17
 *          |                |		            |                |	      
 *          |     TOP        |		            |     BOT        |	      
 *          |                |		            |                |	      
 *   6 7 8  ----------------- 9 10 11	  18 19 20  -----------------  21 22 23
 * 
 *     |
 *     v J
 *
 */


#include "libxtg.h"
#include "libxtg_.h"


int grd3d_point_in_cell(
			int   ibstart,
			int   kzonly,
			double x,
			double y,
			double z,
			int   nx,
			int   ny,
			int   nz,
			double *p_coor_v,
			double *p_zcorn_v,
			int   *p_actnum_v,
			int   maxrad,
			int   sflag,
			int   *nradsearch,
			int   option,
			int   debug
			)
    
{
    /* locals */
    int i, j, k, ib, inside, irad;
    int i1, i2, j1, j2, k1, k2;
    int istart, jstart, kstart,m;
    double corners[24];
    double polx[5], poly[5];
    char  s[24]="grd3d_point_in_cell";

    xtgverbose(debug);
    xtg_speak(s,2,"Entering <grd3d_point_in_cell>");
    xtg_speak(s,2,"NX NY NZ: %d %d %d", nx, ny, nz);

    xtg_speak(s,2,"IBSTART %d", ibstart);

    if (ibstart<0) ibstart=0;

    if (kzonly>0 && ibstart==0) {
	ibstart=x_ijk2ib(1,1,kzonly,nx,ny,nz,0);
    }

    x_ib2ijk(ibstart,&istart,&jstart,&kstart,nx,ny,nz,0);

    /*
     * Will search in a growing radius around a start point
     * in order to optimize speed
     */

    i1=istart;
    j1=jstart;
    k1=kstart;

    i2=istart;
    j2=jstart;
    k2=kstart;

    for (irad=0; irad<=(maxrad+1); irad++) {

	xtg_speak(s,2,"Search radi %d",irad);
	
	if (irad>0) {
	    i1-=1;
	    i2+=1;
	    j1-=1;
	    j2+=1;
	    k1-=1;
	    k2+=1;
	}
	
	if (sflag>0 && irad>maxrad) {
	    i1=1;
	    i2=nx;
	    j1=1;
	    j2=ny;
	    k1=1;
	    k2=nz;
	}
	
	*nradsearch=irad;

	if (i1<1)  i1=1;
	if (j1<1)  j1=1;
	if (k1<1)  k1=1;
	if (i2>nx) i2=nx;
	if (j2>ny) j2=ny;
	if (k2>nz) k2=nz;
	
	if (kzonly>0) {
	    k1=kzonly;
	    k2=kzonly;
	}

	if (debug>3) {
	    xtg_speak(s,4,"I1 I2  J1 J2  K1 K2  %d %d  %d %d  %d %d", 
		      i1,i2,j1,j2,k1,k2);
	}

	for (k = k1; k <= k2; k++) {
	    for (j = j1; j <= j2; j++) {
		for (i = i1; i <= i2; i++) {
		    
		    if (debug>3) {
			xtg_speak(s,3,"Cell IJK: %d %d %d",i,j,k);
		    }
		    ib=x_ijk2ib(i,j,k,nx,ny,nz,0);
		    /* get the corner for the cell */
		    grd3d_corners(i,j,k,nx,ny,nz,p_coor_v,p_zcorn_v,
			      corners,debug);



		    if (option==0) {
			/* 3D cell */
			inside=x_chk_point_in_cell(x,y,z,corners,1,debug);
			
		    }
		    else {
			/*2D view ...  make a closed polygon in XY 
			  from the corners */
			polx[0]=0.5*(corners[0]+corners[12]);
			poly[0]=0.5*(corners[1]+corners[13]);
			polx[1]=0.5*(corners[3]+corners[15]);
			poly[1]=0.5*(corners[4]+corners[16]);
			polx[2]=0.5*(corners[9]+corners[21]);
			poly[2]=0.5*(corners[10]+corners[22]);
			polx[3]=0.5*(corners[6]+corners[18]);
			poly[3]=0.5*(corners[7]+corners[19]);
			polx[4]=polx[0];
			poly[4]=poly[0];
			
			if (debug>2) {
			    for (m=0;m<5;m++) {
				xtg_speak(s,3,"Corner no %d:  %9.2f   %9.2f ",
					  m+1,polx[m],poly[m]);
			    }
			}


			inside=pol_chk_point_inside((double)x,(double)y,
						    polx,poly,5,debug);
			if (debug>2) {
			    xtg_speak(s,3,"Inside status: %d", inside);
			}
		    }
		    
		    if (inside > 0) {
			xtg_speak(s,2,"Found at IJK: %d %d %d",i,j,k);
			return ib;		
		    }
		    
		} 				   
	    }
	}

	if (i1==1 && i2==nx && j1==1 && j2==ny && k1==1 && k2==nz) break;

    }

    return -1; /* if nothing found */
}



