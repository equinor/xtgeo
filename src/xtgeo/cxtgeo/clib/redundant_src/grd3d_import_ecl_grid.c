/*
 * ############################################################################
 * grd3d_import_ecl_grid.c
 * Basic routines to handle import of 3D grids from Eclipse
 * Note; the treament of undefined cells (especially from RMS export) is
 * difficult. Not sure if this works with GRIDFILE 1?
 * Author: JCR
 * ############################################################################
 * $Id: grd3d_import_eclipse_grid.c,v 1.1 2001/03/14 08:02:29 bg54276 Exp $ 
 * $Source: /d/proj/bg/avresmod/src/gplib/GPLExt/RCS/grd3d_import_eclipse_grid.c,v $ 
 *
 * $Log: grd3d_import_eclipse_grid.c,v $
 * Revision 1.1  2001/03/14 08:02:29  bg54276
 * Initial revision
 *
 * Revision 1.1  2000/12/12 17:24:54  bg54276
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
 * ****************************************************************************
 *                        GRD3D_IMPORT_ECL_GRID
 * ****************************************************************************
 * The format is the Eclipse binary/ascii output; the GRID or FGRID file, 
 * This is a corner-point format.
 *
 * The GRID file has the following look:
 * 'DIMENS  '           3 'INTE'
 *          77          99          15
 * 'RADIAL  '           1 'CHAR'
 * 'FALSE   '
 * 'COORDS  '           7 'INTE'
 *           8           1           1           8           0           0
 *           0
 * 'CORNERS '          24 'REAL'
 *    .41300209E+06    .71332070E+07    .35028701E+04    .41314041E+06
 *    .71331340E+07    .35261799E+04    .41294591E+06    .71331490E+07
 *    .34913301E+04    .41307969E+06    .71330730E+07    .35180801E+04
 *    .41300209E+06    .71332070E+07    .35029500E+04    .41314041E+06
 *    .71331340E+07    .35263501E+04    .41294591E+06    .71331490E+07
 *    .34914500E+04    .41307969E+06    .71330730E+07    .35182900E+04
 * etc
 * For the binary form, the record starts and ends with a 4 byte integer, that
 * says how long the current record is, in bytes.
 *
 * ----------------------------------------------------------------------------
 *
 */   

   
void grd3d_import_ecl_grid (
			    int    mode,
			    int    nxyz,
			    int    *num_active,
			    double *p_coord_v,
			    double *p_zcorn_v,
			    int    *actnum_v,
			    char   *filename,
			    int    debug
			    )
{
    
    /* locals */
    int            nx, ny, nz;
    int            ios=0, reclen;
    int            ix, jy, kz, ia = -1, ib=0, ic;
    int            i, j, k, ibb, ibt, nn, mamode, k1, k2;
    float          *tmp_float_v;
    double         x1, y1, x2, y2, x3, y3, cx, cy;
    double         xmin, xmax, ymin, ymax;
    double         cellcorners_v[nxyz][24];   /* should be possible as this is local */
    double         *tmp_double_v;
    int            *tmp_int_v, *tmp_logi_v;
    char           **tmp_string_v;
    int            max_alloc_int, max_alloc_float, max_alloc_double;
    int   	   max_alloc_char, max_alloc_logi;
    
    /* length of char must include \0 termination? */
    char           cname[9], ctype[5];
    char           s[24]="grd3d_import_ecl_grid";
	
    FILE *fc;

    /* 
     * ========================================================================
     * INITIAL TASKS
     * ========================================================================
     */
    xtgverbose(debug);

    /*
     * Some debug info...
     */
    xtg_speak(s,3,"File type is %d", mode);
    xtg_speak(s,3,"NXYZ is %d", nxyz);

    /* RMS view data
     * ------------------------------------------------------------------------
     * I now need to allocate space for tmp_* arrays
     * The calling Perl routine should know (estimate) the total gridsize 
     * (nxyz).
     * ------------------------------------------------------------------------
     */
    max_alloc_int     = 2*nxyz;
    max_alloc_float   = nxyz;
    max_alloc_double  = nxyz;
    max_alloc_char    = nxyz;
    max_alloc_logi    = nxyz;

    tmp_int_v=calloc(max_alloc_int, sizeof(int));
    xtg_speak(s,3,"Allocated tmp_int_v ... OK");
    tmp_float_v=calloc(max_alloc_float, sizeof(float));
    xtg_speak(s,3,"Allocated tmp_float_v ... OK");
    tmp_double_v=calloc(max_alloc_double, sizeof(double));    
    xtg_speak(s,3,"Allocated tmp_double_v ... OK");
    /* the string vector is 2D */
    tmp_string_v=calloc(max_alloc_char, sizeof(char *)); 
    for (i=0; i<nxyz; i++) tmp_string_v[i]=calloc(9, sizeof(char));
    tmp_logi_v=calloc(max_alloc_logi, sizeof(int));

    
    xtg_speak(s,2,"Opening %s",filename);

    fc=fopen(filename,"r");

    xtg_speak(s,2,"Finish opening %s",filename);

    xtg_speak(s,3,"IOS is %d",ios);

    /* initial settings of data */
    xmin=VERYLARGEFLOAT;
    ymin=VERYLARGEFLOAT;
    xmax=-1*VERYLARGEFLOAT;
    ymax=-1*VERYLARGEFLOAT;



    /* 
     * ========================================================================
     * READ RECORDS AND COLLECT NECESSARY STUFF
     * ========================================================================
     */

    /* initialise */
    mamode=0;
    x1=0.0; y1=0.0; x2=0.0; y2=0.0; x3=0.0; y3=0.0;
    nx=1;ny=1;nz=1;
    ix=1;jy=1;kz=1;

    while (ios == 0) {
	
	if (mode == 0) {
	    xtg_speak(s,4,"Reading binary record...");
	    ios=u_read_ecl_bin_record (
				       cname,
				       ctype,
				       &reclen,
				       max_alloc_int,
				       max_alloc_float,
				       max_alloc_double,
				       max_alloc_char,
				       max_alloc_logi,
				       tmp_int_v,
				       tmp_float_v,
				       tmp_double_v,
				       tmp_string_v,
				       tmp_logi_v,
				       fc,
				       debug
				       );
	}
	else{
	    ios=u_read_ecl_asc_record (
				       cname,
				       ctype,
				       &reclen,
				       tmp_int_v,
				       tmp_float_v,
				       tmp_double_v,
				       tmp_string_v,
				       tmp_logi_v,
				       fc,
				       debug
				       );
	}
	    

	if (ios != 0) break;
	xtg_speak(s,3,"Record read for file type %d",mode);


	if (strcmp(cname,"DIMENS  ")==0) {
	    xtg_speak(s,2,"Reading DIMENS values...");
	    nx=tmp_int_v[0];
	    ny=tmp_int_v[1];
	    nz=tmp_int_v[2]; 
	    xtg_speak(s,2,"Found DIMENS %d x %d x %d = %d",
		      nx,
		      ny,
		      nz,
		      nx * ny * nz);
	    
	}

	/*
	 * MAPAXES format is 6 numbers:
	 * xcoord_endof_yaxis ycoord_endof_yaxis xcoord_origin ycoord_origin
	 * xcoord_endof_xaxis ycoord_endof_xaxis
	 * ie
	 * x1 y1 x2 y2 x3 y3 (point 1 is on the new Y axis, point 2 in origo, 
	 * point 3 on new X axis)
	 */


	else if (strcmp(cname,"MAPAXES ")==0) {

	    xtg_speak(s,2,"Reading MAPAXES values...");
	    x1=tmp_float_v[0];
	    y1=tmp_float_v[1];
	    x2=tmp_float_v[2];
	    y2=tmp_float_v[3];
	    x3=tmp_float_v[4];
	    y3=tmp_float_v[5];
	
	    mamode=1;
	}
    

	else if (strcmp(cname,"COORDS  ")==0) {
	    xtg_speak(s,3,"Reading COORDS...");
	    ix=tmp_int_v[0];
	    jy=tmp_int_v[1];
	    kz=tmp_int_v[2]; 
	
	    ib=tmp_int_v[3]-1; /* ib in XTGeo has base zero */
	
	    /* determine if active cell:*/
	    if (tmp_int_v[4] > 0) {
		ia++;
		actnum_v[ib]=1;
	    }
	    else{
		actnum_v[ib]=0;
		/* gridfiletype=2; flag that grid is a GRIDFILE 2 file*/
	    }
	    
	}

	/*
	 * All CORNERS are stored in ONE long 1D array
	 */

	else if (strcmp(cname,"CORNERS ")==0) {
	    xtg_speak(s,3,"Reading CORNERS...");

	    /* store all cells in a big array, and extract COORD and ZCORN afterwards */
	    /*also find min and max values for X and Y */
	    ib=x_ijk2ib(ix,jy,kz,nx,ny,nz,0);
	    for (nn=0; nn<24; nn++) {
		cellcorners_v[ib][nn]=tmp_float_v[nn];
		
		if (nn==0 || nn==3 || nn==6 || nn==9 || nn==12 || nn==15 || nn==18 || nn==21) {
		    if (tmp_float_v[nn]<xmin) xmin=tmp_float_v[nn];
		    if (tmp_float_v[nn]>xmax) xmax=tmp_float_v[nn];
		}

		if (nn==1 || nn==4 || nn==7 || nn==10 || nn==13 || nn==16 || nn==19 || nn==22) {
		    if (tmp_float_v[nn]<ymin) ymin=tmp_float_v[nn];
		    if (tmp_float_v[nn]>ymax) ymax=tmp_float_v[nn];
		}

	    }	    
	    
	}

	else {
	    xtg_speak(s,2,"Reading (and skipping) record: %s\n", cname);
	}
	
    }
	    
    *num_active=ia+1;
    xtg_speak(s,2,"Number of active cells: %d %d", *num_active, ia);

    if (debug>3){
	for (ib=0;ib<nxyz;ib++) {
	    x_ib2ijk(ib,&i,&j,&k,nx,ny,nz,0);
	    xtg_speak(s,4,"======= IJK %d %d %d =======",i,j,k);
	    for (ic=0;ic<24;ic++) {
		xtg_speak(s,4,"CELL %f",cellcorners_v[ib][ic]);
	    }
	    xtg_speak(s,4,"=========================");
	}
    }


    /* extract the COORD lines for all cells (active or inactive) */
    for (j=1; j<=ny; j++) {
	for (i=1; i<=nx; i++) {

	    /* top: */
	    k=1;
	    ib=x_ijk2ib(i,j,k,nx,ny,nz,0);

	    p_coord_v[6*((j-1)*(nx+1)+i-1)+0]=cellcorners_v[ib][0];
	    p_coord_v[6*((j-1)*(nx+1)+i-1)+1]=cellcorners_v[ib][1];
	    p_coord_v[6*((j-1)*(nx+1)+i-1)+2]=cellcorners_v[ib][2];
	    
	    if (i==nx) {
		p_coord_v[6*((j-1)*(nx+1)+i-0)+0]=cellcorners_v[ib][3];
		p_coord_v[6*((j-1)*(nx+1)+i-0)+1]=cellcorners_v[ib][4];
		p_coord_v[6*((j-1)*(nx+1)+i-0)+2]=cellcorners_v[ib][5];
	    }
	    if (j==ny) {
		p_coord_v[6*((j-0)*(nx+1)+i-1)+0]=cellcorners_v[ib][6];
		p_coord_v[6*((j-0)*(nx+1)+i-1)+1]=cellcorners_v[ib][7];
		p_coord_v[6*((j-0)*(nx+1)+i-1)+2]=cellcorners_v[ib][8];

		if (i==nx) {
		    p_coord_v[6*((j-0)*(nx+1)+i-0)+0]=cellcorners_v[ib][9];
		    p_coord_v[6*((j-0)*(nx+1)+i-0)+1]=cellcorners_v[ib][10];
		    p_coord_v[6*((j-0)*(nx+1)+i-0)+2]=cellcorners_v[ib][11];
		}
	    }


	    /* base: */
	    k=nz;
	    ib=x_ijk2ib(i,j,k,nx,ny,nz,0);

	    p_coord_v[6*((j-1)*(nx+1)+i-1)+3]=cellcorners_v[ib][12];
	    p_coord_v[6*((j-1)*(nx+1)+i-1)+4]=cellcorners_v[ib][13];
	    p_coord_v[6*((j-1)*(nx+1)+i-1)+5]=cellcorners_v[ib][14];
	    
	    if (i==nx) {
		p_coord_v[6*((j-1)*(nx+1)+i-0)+3]=cellcorners_v[ib][15];
		p_coord_v[6*((j-1)*(nx+1)+i-0)+4]=cellcorners_v[ib][16];
		p_coord_v[6*((j-1)*(nx+1)+i-0)+5]=cellcorners_v[ib][17];
	    }
	    if (j==ny) {
		p_coord_v[6*((j-0)*(nx+1)+i-1)+3]=cellcorners_v[ib][18];
		p_coord_v[6*((j-0)*(nx+1)+i-1)+4]=cellcorners_v[ib][19];
		p_coord_v[6*((j-0)*(nx+1)+i-1)+5]=cellcorners_v[ib][20];
	    
		if (i==nx) {
		    p_coord_v[6*((j-0)*(nx+1)+i-0)+3]=cellcorners_v[ib][21];
		    p_coord_v[6*((j-0)*(nx+1)+i-0)+4]=cellcorners_v[ib][22];
		    p_coord_v[6*((j-0)*(nx+1)+i-0)+5]=cellcorners_v[ib][23];
		}
	    }

	}	    

    }	   		      
		      
    /* 
     * Extract (and overwrite!) the COORD lines for _active: cells only... 
     * The reason for this is that some pillar for INACTIVE cells may be unprecise and misplaced;
     * here i only take cell stacks with one or more active cells into consideration *
     */
    for (j=1; j<=ny; j++) {
	for (i=1; i<=nx; i++) {

	    k1=0;
	    k2=0;

 	    for (k=1; k<=nz; k++) {
		ib=x_ijk2ib(i,j,k,nx,ny,nz,0);
		if (actnum_v[ib]==1 && k1 == 0) {
		    k1=k;
		}
	    }

		    
	    for (k=nz; k>=1; k--) {
		ib=x_ijk2ib(i,j,k,nx,ny,nz,0);
		if (actnum_v[ib]==1 && k2 == 0) {
		    k2=k;
		}
	    }

		    

	    /* top: */
	    ib=x_ijk2ib(i,j,k1,nx,ny,nz,0);
	    
	    if (actnum_v[ib]==1) {
		
		if (i==1 && j==2) {
		    for (nn=0;nn<24;nn++) {
			xtg_speak(s,1,"CELL 1,2,1: coord no. %d is %e",nn,cellcorners_v[ib][nn]);
		    }
		}
		
		p_coord_v[6*((j-1)*(nx+1)+i-1)+0]=cellcorners_v[ib][0];
		p_coord_v[6*((j-1)*(nx+1)+i-1)+1]=cellcorners_v[ib][1];
		p_coord_v[6*((j-1)*(nx+1)+i-1)+2]=cellcorners_v[ib][2];
		
		p_coord_v[6*((j-1)*(nx+1)+i-0)+0]=cellcorners_v[ib][3];
		p_coord_v[6*((j-1)*(nx+1)+i-0)+1]=cellcorners_v[ib][4];
		p_coord_v[6*((j-1)*(nx+1)+i-0)+2]=cellcorners_v[ib][5];

		p_coord_v[6*((j-0)*(nx+1)+i-1)+0]=cellcorners_v[ib][6];
		p_coord_v[6*((j-0)*(nx+1)+i-1)+1]=cellcorners_v[ib][7];
		p_coord_v[6*((j-0)*(nx+1)+i-1)+2]=cellcorners_v[ib][8];
		    
		p_coord_v[6*((j-0)*(nx+1)+i-0)+0]=cellcorners_v[ib][9];
		p_coord_v[6*((j-0)*(nx+1)+i-0)+1]=cellcorners_v[ib][10];
		p_coord_v[6*((j-0)*(nx+1)+i-0)+2]=cellcorners_v[ib][11];

	    }

	    /* base: */
	    ib=x_ijk2ib(i,j,k2,nx,ny,nz,0);
	    
	    if (actnum_v[ib]==1) {
		
		p_coord_v[6*((j-1)*(nx+1)+i-1)+3]=cellcorners_v[ib][12];
		p_coord_v[6*((j-1)*(nx+1)+i-1)+4]=cellcorners_v[ib][13];
		p_coord_v[6*((j-1)*(nx+1)+i-1)+5]=cellcorners_v[ib][14];
		
		p_coord_v[6*((j-1)*(nx+1)+i-0)+3]=cellcorners_v[ib][15];
		p_coord_v[6*((j-1)*(nx+1)+i-0)+4]=cellcorners_v[ib][16];
		p_coord_v[6*((j-1)*(nx+1)+i-0)+5]=cellcorners_v[ib][17];
		
		p_coord_v[6*((j-0)*(nx+1)+i-1)+3]=cellcorners_v[ib][18];
		p_coord_v[6*((j-0)*(nx+1)+i-1)+4]=cellcorners_v[ib][19];
		p_coord_v[6*((j-0)*(nx+1)+i-1)+5]=cellcorners_v[ib][20];
		
		p_coord_v[6*((j-0)*(nx+1)+i-0)+3]=cellcorners_v[ib][21];
		p_coord_v[6*((j-0)*(nx+1)+i-0)+4]=cellcorners_v[ib][22];
		p_coord_v[6*((j-0)*(nx+1)+i-0)+5]=cellcorners_v[ib][23];
	    }	       	       	       
	
	}
    }	   		      
		      

    /* ZCORN values: */

    for (kz=1; kz<=nz; kz++) {
	for (jy=1; jy<=ny; jy++) {
	    for (ix=1; ix<=nx; ix++) {

		/* cell and cell below*/
		ibt=x_ijk2ib(ix,jy,kz,nx,ny,nz+1,0);
		ibb=x_ijk2ib(ix,jy,kz+1,nx,ny,nz+1,0);
		ib=x_ijk2ib(ix,jy,kz,nx,ny,nz,0);
		
		p_zcorn_v[4*ibt + 1*1 - 1]=cellcorners_v[ib][2];
		p_zcorn_v[4*ibt + 1*2 - 1]=cellcorners_v[ib][5];
		p_zcorn_v[4*ibt + 1*3 - 1]=cellcorners_v[ib][8];
		p_zcorn_v[4*ibt + 1*4 - 1]=cellcorners_v[ib][11];
		
		p_zcorn_v[4*ibb + 1*1 - 1]=cellcorners_v[ib][14];
		p_zcorn_v[4*ibb + 1*2 - 1]=cellcorners_v[ib][17];
		p_zcorn_v[4*ibb + 1*3 - 1]=cellcorners_v[ib][20];
		p_zcorn_v[4*ibb + 1*4 - 1]=cellcorners_v[ib][23];
	    }
	}
    }



    /* convert from MAPAXES, if present */
    if (mamode==1) {
	xtg_speak(s,2,"Conversion via MAPAXES...");
	xtg_speak(s,3,"Using xmin xmax, ymin, ymax... %f  %f    %f  %f",xmin,xmax,ymin,ymax);
	for (ib=0; ib<(nx+1)*(ny+1)*6; ib=ib+3) {
	    cx = p_coord_v[ib];
	    cy = p_coord_v[ib+1];
	    x_mapaxes(mamode,&cx,&cy,x1,y1,x2,y2,x3,y3,xmin,xmax,ymin,ymax,0,debug);
	    p_coord_v[ib]   = cx;
	    p_coord_v[ib+1] = cy;
	}
	xtg_speak(s,2,"Conversion via MAPAXES... DONE");
	    
       
    }	      


    /* free allocated space */
    xtg_speak(s,2,"Freeing tmp pointers");
    xtg_speak(s,3,"Freeing tmp_int_v ...");
    free(tmp_int_v);

    xtg_speak(s,3,"Freeing tmp_float_v ...");
    free(tmp_float_v);
    xtg_speak(s,3,"Freeing tmp_double_v ...");
    free(tmp_double_v);

    /*
      xtg_speak(s,3,"Freeing tmp_int_v ...");
      for (i=0; i<nxyz; i++) free(tmp_string_v[i]);
      free(tmp_string_v); */
    xtg_speak(s,3,"Freeing tmp_logi_v ...");
    free(tmp_logi_v);
    xtg_speak(s,2,"Leaving routine ...");
}

