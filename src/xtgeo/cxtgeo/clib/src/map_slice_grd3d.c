/*
 * ############################################################################
 * Name: map_slice_grd3d
 * By:   JCR
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
 * Input a depth map, returns an map array that holds the ib's for the 3D grid:
 * p_ib_v[mapnode] = ib_cell_number; otherwise undefined
 * ############################################################################
 */

void map_slice_grd3d (
		      int   nx,
		      int   ny,
		      int   nz,
		      double *p_coord_v,
		      double *p_zcorn_v,
		      int   *p_actnum_v,
		      int   mx,
		      int   my,
		      double xmin,
		      double xstep,
		      double ymin,
		      double ystep,
		      double *p_zval_v,   /* input map with Z values */
		      int   *p_ib_v,     /* the array that holds the ib...*/
		      int   option,
		      int   debug
		      )
     
{
    /* locals */
    char  s[24]="map_slice_grd3d"; 
    int   i, j, k, ib, ibm, ic, im, jm, mapdefined, ok, numcount;
    int   immin, immax, jmmin, jmmax;
    double x, y, z, xgmin, xgmax, ygmin, ygmax, zgmin, zgmax;
    double zmmin, zmmax;
    double c[24]; /* corners */
  
    xtgverbose(debug);

    xtg_speak(s,2,"Entering this routine ...");
    
    xtg_speak(s,3,"NX NY NZ: %d %d %d", nx, ny, nz);
    xtg_speak(s,3,"MX and MY is: %d %d", mx, my);

    numcount=0;
    /* work with every 3D cell */
    for (i=1;i<=nx;i++) {
	xtg_speak(s,2,"Working with column %d of %d ...",i,nx);
	for (j=1;j<=ny;j++) {
	    for (k=1;k<=nz;k++) {
	
		/* find if cell is active, or not */
		ib=x_ijk2ib(i,j,k,nx,ny,nz,0);
		if (p_actnum_v[ib]==1) {

		    /* now find the cell corners */
		    grd3d_corners(i,j,k,nx,ny,nz,p_coord_v,p_zcorn_v,
				  c,debug);

		    
		    /* find the min and max of x, y, z of the corners */
		    xgmin=VERYLARGEFLOAT; xgmax=-1*VERYLARGEFLOAT;
		    ygmin=VERYLARGEFLOAT; ygmax=-1*VERYLARGEFLOAT;
		    zgmin=VERYLARGEFLOAT; zgmax=-1*VERYLARGEFLOAT;
		    for (ic=0;ic<23;ic+=3) { /* 8 corners */
			x=c[ic+0]; y=c[ic+1]; z=c[ic+2];
			if (x<xgmin) xgmin=x; if (x>xgmax) xgmax=x; 
			if (y<ygmin) ygmin=y; if (y>ygmax) ygmax=y; 
			if (z<zgmin) zgmin=z; if (z>zgmax) zgmax=z; 
		    }			
		    
		    /* now find the immin, immax, jmmin, jmmax for map */
		    /* note: immin=1 at xstart; as standard map */
		    immin = (int) (((xgmin-xmin)/xstep)+2);
		    immax = (int) (((xgmax-xmin)/xstep)+1);
		    jmmin = (int) (((ygmin-ymin)/ystep)+2);
		    jmmax = (int) (((ygmax-ymin)/ystep)+1);

		    /* some boundary tests, always needed */
 		    if (immin<1) immin=1; if (immax<immin) immax=immin;
		    if (immax>mx) immax=mx; if (immin>immax) immin=immax;
		    if (jmmin<1) jmmin=1; if (jmmax<jmmin) jmmax=jmmin;
		    if (jmmax>my) jmmax=my; if (jmmin>jmmax) jmmin=jmmax;

		    /* find the min and max of map within area */
		    mapdefined=0;
		    zmmin=VERYLARGEFLOAT; zmmax=-1*VERYLARGEFLOAT;
		    for (im=immin;im<=immax;im++) {
			for (jm=jmmin;jm<=jmmax;jm++) {
			    ibm=x_ijk2ib(im,jm,1,mx,my,1,0);
			    z=p_zval_v[ibm];
			    if (z < UNDEF_MAP_LIMIT) {
				if (z<zmmin) zmmin=z;
				if (z>zmmax) zmmax=z;
				mapdefined=1;
			    }
			}
		    }

		    /* 
		     * now, if map is defined AND z of map is 
		     * within z of grid cell, then a corner lookup
		     * will be done...
		     */
		    
		    if (mapdefined==1) {
			if (zgmin<=zmmin && zgmax>=zmmax) {
			    for (im=immin;im<=immax;im++) {
				for (jm=jmmin;jm<=jmmax;jm++) {
				    ibm=x_ijk2ib(im,jm,1,mx,my,1,0);
				    x=xmin + (im-1)*xstep;
				    y=ymin + (jm-1)*ystep;
				    z=p_zval_v[ibm];
				    
				    ok=x_chk_point_in_cell(x,y,z,c,1,debug);
				    
				    if (ok>0) {
					p_ib_v[ibm]=ib;
					numcount++;
				    }
				}
			    }
			}
		    } /* map has defined points */


		}
	    }
	}
    }
    xtg_speak(s,2,"Number of defined map points should be %d",numcount);
    xtg_speak(s,2,"Exiting <map_slice_grd3d>");
}
