/*
 * ############################################################################
 * map_wiener_from_grd3d
 * ############################################################################
 * $Id: map_wiener_from_grd3d.c,v 1.1 2001/03/14 08:02:29 bg54276 Exp $ 
 * $Source: /d/proj/bg/avresmod/src/gplib/GPLExt/RCS/map_wiener_from_grd3d.c,v $ 
 *
 * $Log: map_wiener_from_grd3d.c,v $
 * Revision 1.1  2001/03/14 08:02:29  bg54276
 * Initial revision
 *
 *
 *
 *
 * ############################################################################
 */


#include "libxtg.h"
#include "libxtg_.h"

/*
 * ############################################################################
 * Extract a map from a 3D grid on constant Z level
 * ############################################################################
 */

void map_wiener_from_grd3d (
			      double z,
			      int nx,
			      int ny,
			      int nz,
			      double *p_coord_v,
			      double *p_zcorn_v,
			      char  *ptype,
			      int   *p_int_v,
			      double *p_double_v,
			      int mx,
			      int my,
			      double xmin,
			      double xstep,
			      double ymin,
			      double ystep,
			      double *p_map_v,
			      int   debug
			      )

{
    /* locals */
    int ib=0, ibfound=-1, ic, iq, ibm;
    int i, j, k, mc, inside, jb;
    int i1, i2, j1, j2, k1, k2, imm, imdir;
    int *ibz;
    int im, jm;
    double x, y, cmin, cmax;
    double c[9], corners[24];
    char s[24]; 
  
    strcpy(s,"map_wiener_from_grd3d");


    xtgverbose(debug);

    xtg_speak(s,2,"Entering <map_wiener_from_grd3d>");
    
    xtg_speak(s,3,"NX NY NZ: %d %d %d", nx, ny, nz);
    xtg_speak(s,3,"MX and MY is: %d %d", mx, my);
    xtg_speak(s,3,"XMIN: %13.2f", xmin);
    xtg_speak(s,3,"Parameter type is: <%s>", ptype);

    /*
     * ------------------------------------------------------------------------
     * Scanning ...
     * Will collect all possible cells indexes (ib's) in an own array; i.e.
     * cells that have top edge above z and bottom edge below z
     * ------------------------------------------------------------------------
     */
    mc=0;
    if ((ibz=calloc(nx*ny*nz,sizeof(int))) != NULL) {
	xtg_speak(s,3,"Allocating ...");
    }
    else{
	xtg_error(s,"Cannot allocate ibz!");
    }	    
    
    xtg_speak(s,2,"Scanning grid ...");
    for (k=1;k<=nz;k++) {
	for (j=1;j<=ny;j++) {
	    for (i=1;i<=nx;i++) {
		ib=x_ijk2ib(i,j,k,  nx,ny,nz+1,0);
		iq=x_ijk2ib(i,j,k+1,nx,ny,nz+1,0);
		
		for (ic=1;ic<=8;ic++) {
		    if (ic <= 4) {
			c[ic]=p_zcorn_v[4*ib + ic - 1];
		    }
		    else{
			c[ic]=p_zcorn_v[4*iq + ic - 4 - 1];
		    }
		}
		cmin=VERYLARGEFLOAT;
		for (ic=1;ic<=4;ic++) {
		    if (c[ic]<cmin) cmin=c[ic];
		}
				       
		cmax=VERYSMALLFLOAT;
		for (ic=5;ic<=8;ic++) {
		    if (c[ic]>cmax) cmax=c[ic];
		}
		
		if (cmin <= z && cmax >= z) {
		    ibz[mc++]=ib;
		}
		
		if (i==36 && j==54 && k==113) {
		    for (ic=1;ic<=8;ic++) {
			printf("Ci is %f\n",c[ic]);
		    }
		    printf("CMAX and CMIN %f %f\n: ",cmax,cmin);
		    printf("Cell is IB and IBZ %d %d\n: ",ib, ibz[mc-1]);
		}
	    }
	}
    }		
    xtg_speak(s,2,"Scanning grid ...DONE!");
    xtg_speak(s,2,"%d of total %d cells were selected", mc-1, nx*ny*nz);
    imdir=1;
    for (jm=1;jm<=my;jm++) { 
	xtg_speak(s,2,"Working with map column: %d of %d",jm, my);
	for (im=1;im<=mx;im++) { 
	    xtg_speak(s,3,"Working with map row: %d",im);

	    /* for smarter search, the i counter is "snaking" along */
	    if (imdir==1) {
		imm=im;
	    }
	    else {
		imm=mx-im+1;
	    }

	    ibm=x_ijk2ib(imm,jm,1,mx,my,1,0);

	    if ((p_map_v[ibm] > (UNDEF_MAP+0.1)) ||
		(p_map_v[ibm] < (UNDEF_MAP-0.1))) {
		

		x=xmin + (imm-1)*xstep;
		y=ymin + (jm-1)*ystep;

		/*
		 * Use the previous found value to guess that the next is
		 * close. This may speed up execution alot. This is also
		 * the reason for the "snaking" in im
		 */

		
		x_ib2ijk(ibfound,&i,&j,&k,nx,ny,nz,0);
		i1=i-1; i2=i+1;
		j1=j-1; j2=j+1;
		k1=k-1; k2=k+1;
		if (i1<1) i1=1;	if (i2>nx) i2=nx;
		if (j1<1) j1=1;	if (j2>ny) j2=ny;
		if (k1<1) k1=1;	if (k2>nz) k2=nz;

		ibfound=-1;

		for (k=k1;k<=k2;k++) {
		    for (j=j1;j<=j2;j++) {
			for (i=i1;i<=i2;i++) {
			    ib=x_ijk2ib(i,j,k,nx,ny,nz,0);
			    grd3d_corners(i,j,k,nx,ny,nz,p_coord_v,p_zcorn_v,
					  corners,debug);
			    inside=x_chk_point_in_cell(x,y,z,corners,1,debug);
			    if (inside > 0) {
				xtg_speak(s,3,"Found at IJK: %d %d %d",i,j,k);
				ibfound=ib;
			    }
			}
		    }
		}


		/*
		 * In case the point is not found in the closest area, the
		 * "layer" (selected in ibz array) is searched
		 */


		if (ibfound == -1) {
		    for (jb=0;jb<mc;jb++){
			
			ib=ibz[jb];
			x_ib2ijk(ib,&i,&j,&k,nx,ny,nz,0);
			grd3d_corners(i,j,k,nx,ny,nz,p_coord_v,p_zcorn_v,
				      corners,debug);
			
			inside=x_chk_point_in_cell(x,y,z,corners,1,debug);
			
			if (inside > 0) {
			    xtg_speak(s,3,"Found at IJK: %d %d %d",i,j,k);
			    ibfound=ib;
			    break;
			}
		    }
		}

		if (strcmp(ptype,"double")==0) {
		    if (ibfound>=0) {
			p_map_v[ibm]=p_double_v[ib];
		    }
		    else{
			p_map_v[ibm]=UNDEF_MAP;
		    }	    
		}
		else{
		    if (ibfound>=0) {
			p_map_v[ibm]=p_int_v[ib]; /*int to double conversion*/
		    }
		    else{
			p_map_v[ibm]=UNDEF_MAP;
		    }	    
		}
	    }
	    if (im==mx) imdir=-1*imdir;
	}
    }
    xtg_speak(s,2,"Exiting <map_wiener_from_grd3d>");
}

