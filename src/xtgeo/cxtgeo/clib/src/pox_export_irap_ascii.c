/*
 * ############################################################################
 * pox_export_irap_ascii.c
 * Exports ASCII IRAP points format
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


void pox_export_irap_ascii (
			       double *p_xp_v,
			       double *p_yp_v,
			       double *p_zp_v,
			       char   *file,
			       int    npoints,
			       int    iflag,
			       int    debug
			       )
{

    FILE    *fc;
    int     j;
    double  x, y, z;
    char    s[24]="pox_export_irap_ascii";

    xtgverbose(debug);

    /*
     *-------------------------------------------------------------------------
     * Open file
     *-------------------------------------------------------------------------
     */
    xtg_speak(s,2,"Entering <pox_export_irap_ascii>");
    xtg_speak(s,2,"Opening file: ...");
    fc=fopen(file,"w");
    if (fc==NULL) {
	xtg_error(s,"Cannot open file");
    }
    xtg_speak(s,2,"Opening file ...DONE");

    /*
     *-------------------------------------------------------------------------
     * Do the export. iflag=0 then export only defined; else export all
     *-------------------------------------------------------------------------
     */

    for (j=0;j<npoints;j++) {
	x=p_xp_v[j]; y=p_yp_v[j]; z=p_zp_v[j];
	if (iflag==0) {
	    if (x<UNDEF_POINT_LIMIT) {
		fprintf(fc,"%14.3f %14.3f %12.3f\n",x,y,z);
	    }
	}
	else{
	    if (x>UNDEF_POINT_LIMIT) {
		x=UNDEF_POINT_IRAP;
		y=UNDEF_POINT_IRAP;
		z=UNDEF_POINT_IRAP;
	    }
	    fprintf(fc,"%14.3f %14.3f %12.3f\n",x,y,z);

	}
    }

    fclose(fc);
    xtg_speak(s,2,"Exit from <pox_export_irap_ascii>");
    return;
}
