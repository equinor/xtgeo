/*
 * ############################################################################
 * grd3d_export_ecl_grid
 * Exporting an Eclipse BINARY grid 
 * Author: JCR
 * ############################################################################
 * $Id: grd3d_export_ecl_grid.c,v 1.1 2001/09/28 20:43:30 bg54276 Exp $ 
 * $Source: /d/proj/bg/avresmod/src/gplib/GPLExt/RCS/grd3d_export_eclipse_grid.c,v $ 
 *
 * $Log: grd3d_export_eclipse_grid.c,v $
 * Revision 1.1  2001/09/28 20:43:30  bg54276
 * Initial revision
 *
 *
 * ############################################################################
 */


#include <string.h>
#include <stdlib.h>
#include "libxtg.h"
#include "libxtg_.h"

/*
 ******************************************************************************
 *                      ECLIPSE GRDECL FILE
 * Mode=0 binary else ASCII
 ******************************************************************************
 * ----------------------------------------------------------------------------
 *
 */   


void grd3d_export_ecl_grid (
			    int     nx,
			    int     ny,
			    int     nz,
			    double   *p_coord_v,
			    double   *p_zcorn_v,
			    int     *p_actnum_v,
			    char    *filename,
			    int     mode,
			    int     debug
			    )

{
    int  i, j, k, ib;
    FILE    *fc;
    double  corners_v[24];
    float   fcorners_v[24];
    char    s[24]="grd3d_export_ecl_grid.c";


    /* pointers */
    int    *int_v, *log_v;
    float  *flt_v;

    double *dou_v;
    char   **str_v;

    xtgverbose(debug);

    xtg_speak(s,2,"==== Entering grd3d_export_eclipse_grid ====");
    /* 
     *-------------------------------------------------------------------------
     * Open file
     *-------------------------------------------------------------------------
     */
    
    xtg_speak(s,2,"Opening Eclipse GRID file...");
    if (mode==0) {
      fc=fopen(filename,"wb"); /* windoze require wb for binary files */
    }
    else{
      fc=fopen(filename,"w");
    }
    
    if (fc == NULL) {
      xtg_error(s,"Cannot open file!");
    }
    xtg_speak(s,2,"Opening file...OK!");

    /* 
     *-------------------------------------------------------------------------
     * Allocating
     *-------------------------------------------------------------------------
     */
    int_v=calloc(8,sizeof(int));
    flt_v=calloc(1,sizeof(float));  /* dummy */
    dou_v=calloc(1,sizeof(double)); /* dummy */
    log_v=calloc(1,sizeof(int));    /* dummy */
    str_v=calloc(1, sizeof(char *));
    str_v[0]=calloc(9, sizeof(char));

    
    int_v[0]=nx;
    int_v[1]=ny;
    int_v[2]=nz;
    

    u_wri_ecl_bin_record(
			 "DIMENS  ",
			 "INTE",
			 3,
			 int_v,
			 flt_v,
			 dou_v,
			 str_v,
			 log_v,
			 fc,
			 debug
			 );
    
    for (k=1;k<=nz;k++) {
	for (j=1;j<=ny;j++) {
	    for (i=1;i<=nx;i++) {
		ib=x_ijk2ib(i,j,k,nx,ny,nz,0);
		int_v[0]=i;
		int_v[1]=j;
		int_v[2]=k;
		int_v[3]++;
		int_v[4]=p_actnum_v[ib];
		int_v[5]=0;		
		int_v[6]=0;
		u_wri_ecl_bin_record(
				     "COORDS  ",
				     "INTE",
				     7,
				     int_v,
				     flt_v,
				     dou_v,
				     str_v,
				     log_v,	
				     fc,
				     debug
				     );
		grd3d_corners(
			      i,
			      j,
			      k,
			      nx,
			      ny,
			      nz,
			      p_coord_v,
			      p_zcorn_v,
			      corners_v,
			      debug
			      );
		
		x_conv_double2float(24,corners_v,fcorners_v,debug);

		u_wri_ecl_bin_record(
				     "CORNERS ",
				     "REAL",
				     24,
				     int_v,
				     fcorners_v,
				     dou_v,
				     str_v,
				     log_v,
				     fc,
				     debug
				     );
				
	    }
	}
    }
    fclose(fc);

    xtg_speak(s,2,"Export finished");

}

