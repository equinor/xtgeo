/*
 * ############################################################################
 * grd3d_export_eclipse_prop
 * Exporting an Eclipse BINARY grid (INIT and RESTART format)
 * Author: JCR
 * ############################################################################
 * $Id: $
 * $Source: $
 *
 * $Log: $
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


void grd3d_export_ecl_prop (
			     int     mode,
			     int     nx,
			     int     ny,
			     int     nz,
			     char    *cname,
			     char    *ctype,
			     int     *int_v,
			     double   *float_v,
			     double  *double_v,
			     char    **string_v,
			     int     *logi_v,
			     int     *p_actnum_v,
			     char    *filename,
			     int     debug
			     )

{
    int      i, j, k, ib, ip;
    FILE     *fc;
    float    *tmp_float_v=NULL;
    int      *tmp_int_v=NULL;
    int      *tmp_logi_v=NULL;
    double   *tmp_double_v=NULL;
    char     **tmp_string_v=NULL;

    char     usecname[ECLNAMELEN], usectype[ECLTYPELEN];
    char     s[24]="grd3d_export_ecl_prop";

    /*
     *-------------------------------------------------------------------------
     * Start
     *-------------------------------------------------------------------------
     */

    xtgverbose(debug);

    xtg_speak(s,2,"Entering routine ...");
    /*
     *-------------------------------------------------------------------------
     * Open file
     *-------------------------------------------------------------------------
     */

    xtg_speak(s,2,"Opening INIT or RESTART file (append)...");
    if (mode==0)  {
      fc=fopen(filename,"ab");
    }
    else{
      fc=fopen(filename,"a");
    }

    if (fc == NULL) {
	xtg_error(s,"Cannot open file!");
    }
    xtg_speak(s,2,"Opening file...OK!");


    /*
     *-------------------------------------------------------------------------
     * Allocate
     *-------------------------------------------------------------------------
     */

    if (strcmp(ctype,"int")==0) tmp_int_v=calloc(nx*ny*nz,sizeof(int));
    if (strcmp(ctype,"byte")==0) tmp_int_v=calloc(nx*ny*nz,sizeof(int));
    if (strcmp(ctype,"float")==0)  tmp_float_v=calloc(nx*ny*nz,sizeof(float));
    if (strcmp(ctype,"double")==0)  tmp_double_v=calloc(nx*ny*nz,sizeof(double));
    if (strcmp(ctype,"bool")==0)  tmp_logi_v=calloc(nx*ny*nz,sizeof(int));
    tmp_string_v=calloc(nx*ny*nz, sizeof(char *));
    for (i=0; i<(nx*ny*nz); i++) tmp_string_v[i]=calloc(9, sizeof(char));

    /*
     *-------------------------------------------------------------------------
     * Ensure valid names and types
     *-------------------------------------------------------------------------
     */
    if (strcmp(ctype,"int")==0) strcpy(usectype,"INTE");
    if (strcmp(ctype,"byte")==0) strcpy(usectype,"INTE");
    if (strcmp(ctype,"float")==0) strcpy(usectype,"REAL");
    if (strcmp(ctype,"double")==0) strcpy(usectype,"DOUB");
    if (strcmp(ctype,"bool")==0) strcpy(usectype,"LOGI");
    if (strcmp(ctype,"char")==0) strcpy(usectype,"CHAR");


    for (i=0;i<8;i++) {
	usecname[i]=cname[i];
    }
    u_eightletter(usecname);

    xtg_speak(s,2,"Name is <%s> and type is <%s>",usecname,usectype);

    ip=0;
    if (strcmp(usecname,"PORV")==0) {
	xtg_speak(s,2,"PORV will be written for all cells ...");
	for (ib=0;ib<nx*ny*nz;ib++)tmp_float_v[ib]=float_v[ib];
    }

    for (k=1;k<=nz;k++) {
	for (j=1;j<=ny;j++) {
	    for (i=1;i<=nx;i++) {
		ib=x_ijk2ib(i,j,k,nx,ny,nz,0);
		if (p_actnum_v[ib]==1){
		    if (strcmp(ctype,"int")==0) tmp_int_v[ip++]=int_v[ib];
		    if (strcmp(ctype,"byte")==0) tmp_int_v[ip++]=int_v[ib];
		    if (strcmp(ctype,"float")==0) tmp_float_v[ip++]=float_v[ib];
		    if (strcmp(ctype,"double")==0) tmp_double_v[ip++]=double_v[ib];
		    if (strcmp(ctype,"bool")==0) tmp_logi_v[ip++]=logi_v[ib];
		}
	    }
	}
    }

    u_wri_ecl_bin_record(
			 usecname,
			 usectype,
			 ip,
			 tmp_int_v,
			 tmp_float_v,
			 tmp_double_v,
			 tmp_string_v,
			 tmp_logi_v,
			 fc,
			 debug
			 );

    xtg_speak(s,3,"Closing file ...");
    fclose(fc);

    if (strcmp(ctype,"int")==0) free(tmp_int_v);
    if (strcmp(ctype,"byte")==0) free(tmp_int_v);
    if (strcmp(ctype,"float")==0)  free(tmp_float_v);
    if (strcmp(ctype,"double")==0)  free(tmp_double_v);
    if (strcmp(ctype,"bool")==0)  free(tmp_int_v);
}
