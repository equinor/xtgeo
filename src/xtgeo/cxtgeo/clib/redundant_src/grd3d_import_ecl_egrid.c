/*
 * ############################################################################
 * grd3d_import_ecl_egrid.c
 * Basic routines to handle import of 3D grids from Eclipse; here EGRID format
 * Author: JRIV
 * ############################################################################
 *
 * ############################################################################
 */


#include <string.h>
#include <stdlib.h>
#include "libxtg.h"
#include "libxtg_.h"

/*
 * ****************************************************************************
 *                        GRD3D_IMPORT_ECL_EGRID
 * ****************************************************************************
 * The format is the Eclipse binary/ascii output; the EGRID or FEGRID file,
 * This is a corner-point format.
 *
 * The EGRID file has the following look:
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


void grd3d_import_ecl_egrid (
			    int    mode,
			    int    nx,
			    int    ny,
			    int    nz,
			    int    *num_active,
			    double *p_coord_v,
			    double *p_zcorn_v,
			    int    *actnum_v,
			    char   *filename,
			    int    debug
			    )
{

    /* locals */
    int            nxyz, ios=0, reclen;
    int            ix, jy, kz, ia = -1, ib=0, ic, kzread;
    int            i, j, k, nn, mamode, kk;
    float          *tmp_float_v;
    double         x1, y1, x2, y2, x3, y3, cx, cy, fvalue1, fvalue2;
    double         xmin, xmax, ymin, ymax;
    double         *tmp_double_v;
    int            *tmp_int_v, *tmp_logi_v;
    char           **tmp_string_v;
    int            max_alloc_int, max_alloc_float, max_alloc_double;
    int   	   max_alloc_char, max_alloc_logi;

    /* length of char must include \0 termination? */
    char           cname[9], ctype[5];
    char           s[24]="grd3d_import_ecl_egrid";

    FILE *fc;

    /*
     * ========================================================================
     * INITIAL TASKS
     * ========================================================================
     */
    xtgverbose(debug);

    nxyz=nx*ny*nz;

    /*
     * Some debug info...
     */
    xtg_speak(s,3,"File type is %d", mode);
    xtg_speak(s,3,"NXYZ is %d", nxyz);

    /*
     * ------------------------------------------------------------------------
     * I now need to allocate space for tmp_* arrays
     * The calling Perl routine should know (estimate) the total gridsize
     * (nx*ny*nz).
     * ------------------------------------------------------------------------
     */

    max_alloc_int     = 2*nxyz;
    max_alloc_float   = 8*nxyz;
    max_alloc_double  = nxyz;
    max_alloc_char    = nxyz;
    max_alloc_logi    = nxyz;


    tmp_int_v=calloc(max_alloc_int, sizeof(int));
    xtg_speak(s,3,"Allocated tmp_int_v ... OK");
    /* float can read ZCORN, which is nxyz*8 */
    tmp_float_v=calloc(max_alloc_float, sizeof(float));
    xtg_speak(s,3,"Allocated tmp_float_v ... OK");
    tmp_double_v=calloc(max_alloc_double, sizeof(double));
    xtg_speak(s,3,"Allocated tmp_double_v ... OK");
    /* the string vector is 2D */
    tmp_string_v = malloc(max_alloc_char * sizeof(char *));
    if (tmp_string_v)
        for (i=0; i<nxyz; i++)
            tmp_string_v[i] =  malloc(9 * sizeof *tmp_string_v[i]) ;

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


	if (strcmp(cname,"ACTNUM  ")==0) {
	    xtg_speak(s,2,"Reading ACTNUM values...");
	    for (ib=0;ib<nxyz;ib++) {
		actnum_v[ib]=tmp_int_v[ib];
		if (actnum_v[ib]==1) ia++;
	    }

	}

	/*
	 * MAPAXES format is 6 numbers:
	 * xcoord_endof_yaxis ycoord_endof_yaxis xcoord_origin ycoord_origin
	 * xcoord_endof_xaxis ycoord_endof_xaxis
	 * ie
	 * x1 y1 x2 y2 x3 y3 (point 1 is on the new Y axis, point 2 in origo, point 3 on new X axis)
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


	else if (strcmp(cname,"COORD   ")==0) {
	    xtg_speak(s,2,"Reading COORD..., RECLEN is %d",reclen);


	    for (nn=0; nn<reclen; nn++) {


		p_coord_v[nn]=tmp_float_v[nn];

		xtg_speak(s,4,"Coordinate is %f",p_coord_v[nn]);

		/* find xmin/xmax/ymin/ymax: */
		if (ix==1) {
		    if (p_coord_v[nn]<xmin) xmin = p_coord_v[nn];
		    if (p_coord_v[nn]>xmax) xmax = p_coord_v[nn];
		    ix=0;
		    jy=1;
		    kz=0;
		}
		else if (jy==1) {
		    if (p_coord_v[nn]<ymin) ymin = p_coord_v[nn];
		    if (p_coord_v[nn]>ymax) ymax = p_coord_v[nn];
		    ix=0;
		    jy=0;
		    kz=1;
		}
		else{
		    ix=1;
		    jy=0;
		    kz=0;
		}
	    }

	}

	else if (strcmp(cname,"ZCORN   ")==0) {
	    xtg_speak(s,2,"Reading ZCORN...");

	    /*
	     * ZCORN: Eclipse has 8 corners pr cell, while XTGeo format
	     * use 4 corners (top of cell) for NZ+1 cell. This may cause
	     * problems if GAPS in GRDECL format (like BRILLD test case)
	     *
	     */

	    ib=0;
	    kzread=0;
	    kk=0;
	    ic=0;
	    for (k=1; k<=2*nz; k++) {

		xtg_speak(s,4,"ZCORN reading %f",tmp_float_v[ic]);


		if (kzread==0) {
		    kzread=1;
		}
		else{
		    kzread=0;
		}
		if (k==2*nz && kzread==0) kzread=1;
		if (kzread==1) {
		    kk+=1;
		    xtg_speak(s,2,"Reading layer: %d", kk);
		}
		for (j=1; j<=ny; j++) {
		    /* "left" cell margin */
		    for (i=1; i<=nx; i++) {
			fvalue1=tmp_float_v[ic++];
			fvalue2=tmp_float_v[ic++];

			ib=x_ijk2ib(i,j,kk,nx,ny,nz+1,0);
			if (kzread==1) {
			    p_zcorn_v[4*ib+1*1-1] = fvalue1;
			    p_zcorn_v[4*ib+1*2-1] = fvalue2;
			}
		    }
		    /* "right" cell margin */
		    for (i=1; i<=nx; i++) {
			fvalue1=tmp_float_v[ic++];
			fvalue2=tmp_float_v[ic++];

			ib=x_ijk2ib(i,j,kk,nx,ny,nz+1,0);
			if (kzread==1) {
			    p_zcorn_v[4*ib+1*3-1] = fvalue1;
			    p_zcorn_v[4*ib+1*4-1] = fvalue2;
			}
		    }
		}
	    }
	}

	else {
	    xtg_speak(s,2,"Reading (and skipping) record: %s\n", cname);
	}

    }

    *num_active=ia+1;
    xtg_speak(s,2,"Number of active ACTIVE cells: %d %d", *num_active, ia);
    xtg_speak(s,2,"Number of ACTIVE cells: %d %d", *num_active, ia);

    xtg_speak(s,3,"(1)... XMIN XMAX YMIN YMAX: %10.2f %10.2f %10.2f %10.2f",xmin,xmax,ymin,ymax);

    /* get min and max of geometry (for use in MAPAXES Tor Barkve method) */
    /* grd3d_minmax_geom(nx,ny,nz,p_coord_v,p_zcorn_v,actnum_v,  */
    /* 		      &xmin, &xmax, &ymin, &ymax, &zmin, &zmax, 0, debug); */

    xtg_speak(s,3,"(2)... XMIN XMAX YMIN YMAX: %10.2f %10.2f %10.2f %10.2f",xmin,xmax,ymin,ymax);

    /* convert from MAPAXES, if present */
    if (mamode==1) {
	xtg_speak(s,2,"Conversion via MAPAXES...");
	xtg_speak(s,3,"Using xmin xmax, ymin, ymax... %f  %f    %f  %f",xmin,xmax,ymin,ymax);
	for (ib=0; ib<(nx+1)*(ny+1)*6; ib=ib+3) {
	    cx = p_coord_v[ib];
	    cy = p_coord_v[ib+1];
	    x_mapaxes(mamode,&cx,&cy,x1,y1,x2,y2,x3,y3,xmin,xmax,ymin,ymax,2,debug);
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


    xtg_speak(s,3,"Freeing tmp_string_v ...");
    for (i=0; i<nxyz; i++) {
        xtg_speak(s,3,"Freeing tmp_string_v elem...");
        if (tmp_string_v[i]) free(tmp_string_v[i]);
    }
    if (tmp_string_v) free(tmp_string_v);

    xtg_speak(s,3,"Freeing tmp_logi_v ...");
    free(tmp_logi_v);
    xtg_speak(s,2,"Leaving routine ...");
}
