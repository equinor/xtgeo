/*
 * ############################################################################
 * grd3d_export_grdecl
 * Exporting an Eclipse ASCII input grid 
 * Author: JCR
 * ############################################################################
 */


#include <string.h>
#include <stdlib.h>
#include "libxtg.h"
#include "libxtg_.h"

/*
 ******************************************************************************
 *                      ECLIPSE GRDECL FILE
 ******************************************************************************
 * ----------------------------------------------------------------------------
 *
 */   


void grd3d_export_grdecl (
			  int     nx,
			  int     ny,
			  int     nz,
			  double  *p_coord_v,
			  double  *p_zcorn_v,
			  int     *p_actnum_v,
			  char    *filename,
			  int     debug
			  )

{
    int  i, j, k, ib, num_cornerlines;
    int  jj;
    FILE *fc;
    char s[24]="grd3d_export_grdecl";

    xtgverbose(debug);

    xtg_speak(s,2,"==== Entering grd3d_export_grdecl ====");
    /* 
     *-------------------------------------------------------------------------
     * Open file
     *-------------------------------------------------------------------------
     */
    
    xtg_speak(s,2,"Opening GRDECL file...");
    fc=fopen(filename,"wb"); /* The b will ensure Unix style ASCII on Windoz */
    if (fc == NULL) {
	xtg_error(s,"Cannot open file!");
    }
    xtg_speak(s,2,"Opening file...OK!");
    
    
    /* 
     *=========================================================================
     * Loop file... It is NOT necessary to do many tests; that should be done
     * by the calling PERL script?
     *=========================================================================
     */
    num_cornerlines=2*3*(nx+1)*(ny+1);
    xtg_speak(s,2,"NX NY NZ and no. of cornerlines: %d %d %d   %d",
	    nx,ny,nz,num_cornerlines);
    
    
    xtg_speak(s,2,"Exporting SPECGRID...");
    fprintf(fc,"SPECGRID\n");
    fprintf(fc," %d %d %d 1 F /\n",nx,ny,nz);
	
    xtg_speak(s,2,"Exporting COORD...");
    fprintf(fc,"COORD\n");
    ib=0;
    for (j=0;j<=ny; j++) {
	for (i=0;i<=nx; i++) {
	fprintf(fc," %15.3f %15.3f %9.3f   %15.3f %15.3f %9.3f\n",
		p_coord_v[ib+0],p_coord_v[ib+1],p_coord_v[ib+2],
		p_coord_v[ib+3],p_coord_v[ib+4],p_coord_v[ib+5]);
	ib=ib+6;
	}
	fprintf(fc,"\n");
    }
    fprintf(fc,"/\n");
	

    /*
     * ZCORN is ordered cycling X fastest, then Y, then Z, for all 
     * 8 corners. XTGeo format is a bit different, having 4 corners
     * pr cell layer as       3_____4
     *                        |     |
     *                        |     |
     *                        |_____|
     *                        1     2
     */



    xtg_speak(s,2,"Exporting ZCORN...");
    fprintf(fc,"ZCORN\n");
    jj=0;
    for (k=1; k<=nz; k++) {
	/* top */
	fprintf(fc,"-- top layer %d\n",k);
	for (j=1; j<=ny; j++) {
	    for (i=1; i<=nx; i++) {
		ib=x_ijk2ib(i,j,k,nx,ny,nz+1,0);
		jj++;
		fprintf(fc,"%10.3f",p_zcorn_v[4*ib+1*1-1]);
		fprintf(fc,"%10.3f",p_zcorn_v[4*ib+1*2-1]);
		if (jj>5) {
		    jj=0;
		    fprintf(fc,"\n");
		}
	    }
	    for (i=1; i<=nx; i++) {
		ib=x_ijk2ib(i,j,k,nx,ny,nz+1,0);
		jj++;
		fprintf(fc,"%10.3f",p_zcorn_v[4*ib+1*3-1]);
		fprintf(fc,"%10.3f",p_zcorn_v[4*ib+1*4-1]);
		if (jj>5) {
		    jj=0;
		    fprintf(fc,"\n");
		}
	    }	    
	}	
	fprintf(fc,"\n");
	/* bottom */
	fprintf(fc,"-- bottom layer %d\n",k);

	for (j=1; j<=ny; j++) {
	    for (i=1; i<=nx; i++) {
		ib=x_ijk2ib(i,j,k+1,nx,ny,nz+1,0);
		jj++;
		fprintf(fc,"%10.3f",p_zcorn_v[4*ib+1*1-1]);
		fprintf(fc,"%10.3f",p_zcorn_v[4*ib+1*2-1]);
		if (jj>5) {
		    jj=0;
		    fprintf(fc,"\n");
		}
	    }
	    for (i=1; i<=nx; i++) {
		ib=x_ijk2ib(i,j,k+1,nx,ny,nz+1,0);
		jj++;
		fprintf(fc,"%10.3f",p_zcorn_v[4*ib+1*3-1]);
		fprintf(fc,"%10.3f",p_zcorn_v[4*ib+1*4-1]);
		if (jj>5) {
		    jj=0;
		    fprintf(fc,"\n");
		}
	    }	    
	}	

	fprintf(fc,"\n\n");
    }
    fprintf(fc,"/\n");
    
    jj=0;
    xtg_speak(s,2,"Exporting ACTNUM...");
    fprintf(fc,"ACTNUM\n");
    ib=0;
    for (i=1;i<=nx*ny*nz; i++) {
	jj++;
	fprintf(fc,"%3d",p_actnum_v[ib++]);
	if (jj==12) {
	    jj=0;
	    fprintf(fc,"\n");
	}
    }
    fprintf(fc,"/\n");
    fclose(fc);

}

