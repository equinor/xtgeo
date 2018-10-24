/*
 * ############################################################################
 * grd3d_split_prop_by_pol.c
 * For discrete property; if a prop value is present both inside an outside a 
 * polygon, the value inside is recoded to a new unique value.
 * Author: J.C. Rivenaes
 * ############################################################################
 */

#include "libxtg.h"
#include "libxtg_.h"

void grd3d_split_prop_by_pol(
			   int    np,
			   double  *p_xp_v,
			   double  *p_yp_v,
			   int    nx,
			   int    ny,
			   int    nz,
			   double  *p_coord_v,
			   double  *p_zcorn_v,
			   int    *p_actnum_v,
			   int    *p_prop_v,
			   double  *p_propfilter_v,
			   double  filterval,
			   int    i1,
			   int    i2,
			   int    j1,
			   int    j2,
			   int    k1,
			   int    k2,
			   int    option,
			   int    debug
			   )
{
    int i, j, k,  ib, np1,  istat, minv, maxv, mm, lastcode;
    int *p_inside_v, *p_outside_v, *p_newcode_v, *p_tmp_v, csize;
    int *p_proxy_v, *p_usemm_v;
    double  xg, yg, zg;
    
    double x, y;
    char s[24]="grd3d_split_prop_by_pol";

    xtgverbose(debug);

    xtg_speak(s,2,"Split prop by polygon <%s>",s);


    if (i1>nx || i2>nx || i1<1 || i2<1 || i1>i2) {
    	xtg_error(s,"Error in I specification. STOP");
    }
    if (j1>ny || j2>ny || j1<1 || j2<1 || j1>j2) {
    	xtg_error(s,"Error in J specification. STOP");
    }
    if (k1>nz || k2>nz || k1<1 || k2<1 || k1>k2) {
    	xtg_error(s,"Error in K specification. STOP");
    }

    /* first collect the min and max */
    minv=UNDEF_INT;
    maxv=minv*-1;

    for (ib=0; ib<nx*ny*nz; ib++) {
    	if (p_prop_v[ib] < UNDEF_INT_LIMIT && p_prop_v[ib] > -999) {
    	    if (p_prop_v[ib]<minv) minv = p_prop_v[ib];
    	    if (p_prop_v[ib]>maxv) maxv = p_prop_v[ib];
    	}
    }
    
    csize=maxv+20;
    xtg_speak(s,2,"Allocate: %d",csize);


    /* now allocate some lists */
    p_inside_v  = calloc(csize,sizeof(int));
    p_outside_v = calloc(csize,sizeof(int));
    p_newcode_v = calloc(csize,sizeof(int));
    p_proxy_v   = calloc(csize,sizeof(int));
    p_usemm_v   = calloc(csize,sizeof(int));
    p_tmp_v     = calloc(nx*ny*nz,sizeof(int));

    /* initialize: */
    for (mm=minv; mm<=maxv; mm++) {
    	p_inside_v[mm]   = 0;
    	p_outside_v[mm]  = 0;
	p_newcode_v[mm]  = 0; 
	p_proxy_v[mm]    = 0; 
	p_usemm_v[mm]    = 1;   /* maye be changed by proxy */ 
    }

    xtg_speak(s,2,"Initializing done, from %d to %d",minv,maxv);

    /* collect/list cells that are inside and/or outside */
    for (k=k1;k<=k2;k++) {
    	xtg_speak(s,2,"Layer is %d",k);
    	for (j=j1;j<=j2;j++) {
    	    for (i=i1;i<=i2;i++) {

    		grd3d_midpoint(i,j,k,nx,ny,nz,p_coord_v,
    			       p_zcorn_v,&xg,&yg,&zg,debug);
		
    		x= xg;
    		y= yg;
    		/* search if XG, YG is present in polygon */
    		istat=0;
    		np1=0;

    		ib=x_ijk2ib(i,j,k,nx,ny,nz,0);

		p_tmp_v[ib]=-1;  /* for storing cells within polygon, -1 is outside*/

    		xtg_speak(s,3,"Midpoint is %f %f for %d %d %d", x,y, i, j, k);

    		/* will return 2 if point is inside, 1 on edge, and 0 outside. -1 if undeterm */
    		istat=pol_chk_point_inside(
    					   x,
    					   y,
    					   p_xp_v,
    					   p_yp_v,
    					   np,
    					   debug
    					   );

    		if (p_prop_v[ib] < UNDEF_INT_LIMIT && p_prop_v[ib]>-999) {
    		    if (istat>0) {
    			p_inside_v[p_prop_v[ib]] += 1;
			p_tmp_v[ib]=p_prop_v[ib];
    		    }
    		    else{
    			p_outside_v[p_prop_v[ib]] += 1;
    		    }
    		}

		/* 
		 *if propfilter (or proxy) is enables, only those 
		 * values shall be collected 
		 */

		if (filterval != -9999) {
		    if (p_propfilter_v[ib] == filterval) {
			p_proxy_v[p_prop_v[ib]]=1;
		    }
		}
    	    }
    	}
    }


    xtg_speak(s,2,"Collecting done");

    /* the next step is to compare the lists, and renumber cells within the polygon */
    lastcode=maxv;

    for (mm=minv; mm<=maxv; mm++) {
    	xtg_speak(s,2,"Number %d has inside %d and outside %d", mm, p_inside_v[mm], p_outside_v[mm]);
	if (p_inside_v[mm]>0 && p_outside_v[mm]>0) {
	    xtg_speak(s,2,"Code %d may need splitting",mm);
	    if (filterval != -9999) {
		if (p_proxy_v[mm]==1) {
		    p_newcode_v[mm]=lastcode+1;
		    lastcode=lastcode+1;
		    xtg_speak(s,2,"...proxy is active and %d needs splitting",mm);
		}
		else{
		    p_usemm_v[mm]=0;
		    xtg_speak(s,2,"...proxy is active and found that %d needs no splitting",mm);
		}		    
	    }
	    else{
		p_newcode_v[mm]=lastcode+1;
		lastcode=lastcode+1;
		xtg_speak(s,2,"...proxy is not active and %d needs splitting",mm);
	    }
	}
    }


    xtg_speak(s,2,"Recoding...",mm);

   
    /* loop again ... */
    for (k=k1;k<=k2;k++) {
	for (j=j1;j<=j2;j++) {
	    for (i=i1;i<=i2;i++) {
		
    		ib=x_ijk2ib(i,j,k,nx,ny,nz,0);
		/* check if cell value is inside the polygon */
		for (mm=minv; mm<=maxv; mm++) {
		    if (p_inside_v[mm]>0 && p_outside_v[mm]>0 && p_usemm_v[mm]==1) {
			if (p_tmp_v[ib] > -1 && p_tmp_v[ib] == mm) {
			    p_prop_v[ib] = p_newcode_v[mm];
			}
		    }
		    
    		}
    	    }
    	}
    }

    xtg_speak(s,2,"Recoding... DONE",mm);

    free(p_inside_v);
    free(p_outside_v);
    free(p_newcode_v);
    free(p_tmp_v);
    free(p_proxy_v);
    free(p_usemm_v);

    xtg_speak(s,2,"Split prop by polygon <%s> done!",s);

}



