/*
 * ############################################################################
 * grd3d_export_eclipse_pstart
 * Exporting an Eclipse BINARY grid (INIT and RESTART format) header
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


void grd3d_export_ecl_pstart (
			      int     mode,
			      int     nx,
			      int     ny,
			      int     nz,
			      int     *p_actnum_v,
			      char    *filename,
			      int     debug
			      )

{
    int      i;
    int      nact;
    FILE     *fc;
    float    *tmp_float_v;
    int      *tmp_int_v, *tmp_logi_v;
    double   *tmp_double_v;
    char     **tmp_string_v;
    char     s[24]="grd3d_export_ecl_pstart";
    char     ftype[3];
    /* 
     *-------------------------------------------------------------------------
     * Start
     *-------------------------------------------------------------------------
     */

    xtgverbose(debug);

    xtg_speak(s,2,"==== Entering grd3d_export_eclipse_pstart ====");
    /* 
     *-------------------------------------------------------------------------
     * Open file
     *-------------------------------------------------------------------------
     */
    
    xtg_speak(s,2,"Opening INIT or RESTART file...");
    
    strcpy(ftype,"w");
    if (mode==0) strcpy(ftype,"wb"); /* binary file windoze */
    fc=fopen(filename,ftype);
    if (fc == NULL) {
	xtg_error(s,"Cannot open file!");
    }
    xtg_speak(s,2,"Opening file...OK!");


    /* 
     *-------------------------------------------------------------------------
     * Allocate
     *-------------------------------------------------------------------------
     */
    tmp_int_v    = calloc(ECLINTEHEADLEN,4);
    tmp_float_v  = calloc(1,4); /*dummy?*/
    tmp_double_v = calloc(ECLDOUBHEADLEN,8);
    tmp_logi_v   = calloc(ECLLOGIHEADLEN,4);
    tmp_string_v=calloc(1, sizeof(char *)); 
    tmp_string_v[1]=calloc(9, sizeof(char));
    
    /* 
     *-------------------------------------------------------------------------
     * Populate...
     *-------------------------------------------------------------------------
     */
    nact=0;
    /* calculate number of active cells */
    for (i=0;i<nx*ny*nz;i++){
	if (p_actnum_v[i]==1)nact+=1;
    }

    for (i=0;i<ECLINTEHEADLEN;i++) {
	tmp_int_v[i]=0;
	if (i==2) tmp_int_v[i]=1; /*metric*/
	if (i==8) tmp_int_v[i]=nx;
	if (i==9) tmp_int_v[i]=ny;
	if (i==10) tmp_int_v[i]=nz;
	if (i==11) tmp_int_v[i]=nact;
	if (i==16) tmp_int_v[i]=0;
	if (i==17) tmp_int_v[i]=1;
	if (i==18) tmp_int_v[i]=1;
	if (i==19) tmp_int_v[i]=1;
	if (i==20) tmp_int_v[i]=1;
	if (i==21) tmp_int_v[i]=1;
	if (i==22) tmp_int_v[i]=1;
	if (i==23) tmp_int_v[i]=1;
	if (i==24) tmp_int_v[i]=93;
	if (i==27) tmp_int_v[i]=3;
	if (i==32) tmp_int_v[i]=19;
    }

    for (i=0;i<ECLLOGIHEADLEN;i++) {
	tmp_logi_v[i]=0;            /*false*/
	if (i==2) tmp_logi_v[i]=1; /*metric*/
    }

    xtg_speak(s,3,"Writing INTEHEAD...");
    u_wri_ecl_bin_record(
			 "INTEHEAD",
			 "INTE",
			 95,
			 tmp_int_v,
			 tmp_float_v,
			 tmp_double_v,
			 tmp_string_v,
			 tmp_logi_v,
			 fc,
			 debug
			 );

    xtg_speak(s,3,"Writing LOGIHEAD...");
    u_wri_ecl_bin_record(
			 "LOGIHEAD",
			 "LOGI",
			 15,
			 tmp_int_v,
			 tmp_float_v,
			 tmp_double_v,
			 tmp_string_v,
			 tmp_logi_v,
			 fc,
			 debug
			 );
    xtg_speak(s,3,"Writing DOUBHEADHEAD...");
    u_wri_ecl_bin_record(
			 "DOUBHEAD",
			 "DOUB",
			 1,
			 tmp_int_v,
			 tmp_float_v,
			 tmp_double_v,
			 tmp_string_v,
			 tmp_logi_v,
			 fc,
			 debug
			 );
    
    fclose(fc);

    free(tmp_int_v);
    free(tmp_float_v);
    free(tmp_double_v);
    free(tmp_logi_v);
    free(tmp_string_v[1]);
    free(tmp_string_v);
}

