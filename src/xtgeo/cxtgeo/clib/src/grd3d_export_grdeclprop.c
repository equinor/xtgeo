/*
 * ############################################################################
 * grd3d_export_eclipse_grdeclprop
 * Exporting an Eclipse ASCII input grid property record
 * Eclipse GRDECL has no clue about float/int differences...
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


void grd3d_export_grdeclprop (
			      int     nx,
			      int     ny,
			      int     nz,
			      int     formatspec,
			      char    *propname,
			      double   *p_fprop_v,
			      char    *filename,
			      int     filemode,
			      int     debug
			      )

{
    int   i, j;
    double min, max, avg;
    FILE *fc;
    char uformat[33];
    char s[24]="grd3d_export_grdeclprop";

    xtgverbose(debug);

    /*
     *-------------------------------------------------------------------------
     * Open file
     *-------------------------------------------------------------------------
     */

    xtg_speak(s,2,"Opening GRDECL file...");

    if (filemode==1) {
	/* append */
	fc=fopen(filename,"ab"); /* The b will ensure Unix style ASCII on Windoz */
    }
    else{
	fc=fopen(filename,"wb");
    }

    if (fc == NULL) {
	xtg_error(s,"Cannot open file!");
    }
    xtg_speak(s,2,"Opening file...OK!");


    /*
     *-------------------------------------------------------------------------
     * For proper formatting, some statisitics (not sure about the UNDEF
     * here...)
     *-------------------------------------------------------------------------
     */

    if (formatspec==2) {

	x_basicstats(nx*ny*nz,UNDEF,p_fprop_v,&min,&max,&avg,debug);

	if (max<1) {
	    strcpy(uformat,"%7.6f  ");
	}
	else if (max>10000) {
	    strcpy(uformat,"%12.3f  ");
	}
	else{
	    strcpy(uformat,"%10.4f  ");
	}
    }
    else{
	strcpy(uformat,"%8d  ");
    }

    /*
     *-------------------------------------------------------------------------
     * Write and close file
     *-------------------------------------------------------------------------
     */
    j=0;
    xtg_speak(s,2,"Exporting property %s ...",propname);
    fprintf(fc,"%s\n",propname);
    i=0;
    for (i=0;i<nx*ny*nz; i++) {
	j++;
	if (formatspec==1) { /* integer style */
	    if (p_fprop_v[i] < UNDEF_INT_LIMIT) {
		fprintf(fc,uformat,(int)p_fprop_v[i]);
	    }
	    else{
		fprintf(fc,uformat,(int)UNDEF_ECLINT);
	    }

	}
	else{
	    if (p_fprop_v[i] < UNDEF_LIMIT) {
		fprintf(fc,uformat,p_fprop_v[i]);
	    }
	    else{
		fprintf(fc,uformat,UNDEF_ECLFLOAT);
	    }

	}
	if (j==10) {
	    j=0;
	    fprintf(fc,"\n");
	}
    }
    fprintf(fc,"\n/\n");
    fclose(fc);

}
