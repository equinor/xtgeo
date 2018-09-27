/*
 * ############################################################################
 * pox_import_irap.c
 * Reads Irap CLassic ASCII points format (points, polygons, ... = pox !)
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
 * ****************************************************************************
 *                        Irap points format
 * ****************************************************************************
 * 3.1001877e+03 1.6733066e+04 0.0000000e+00
 * 2.7623450e+03 1.7068479e+04 0.0000000e+00
 * 2.4968096e+03 1.7208236e+04 0.0000000e+00
 * ----------------------------------------------------------------------------
 */

int pox_import_irap (
		     double *p_xp_v,
		     double *p_yp_v,
		     double *p_zp_v,
		     int    npoints,
		     char   *file,
		     int    debug
		     )
{

    FILE    *fc;
    int     iok, j, np;
    double  xd, yd, zd;
    char    s[24];

    strcpy(s,"pox_import_irap");
    xtgverbose(debug);

    /*
     *-------------------------------------------------------------------------
     * Open file
     *-------------------------------------------------------------------------
     */


    xtg_speak(s,2,"Entering <pox_import_irap>");
    xtg_speak(s,2,"Opening file: ...");
    fc=fopen(file,"r");
    if (fc==NULL) {
	xtg_error(s,"Cannot open file");
    }
    xtg_speak(s,2,"Opening file ...DONE");

    xtg_speak(s,3,"NPOINTS is %d", npoints);

    np=0;
    for (j=0;j<npoints;j++) {
	iok=fscanf(fc, "%lf %lf %lf", &xd, &yd, &zd);

	//if (iok != NULL) np++;
	np++;

	xtg_speak(s,3,"Read line: IOK is %d", iok);
	xtg_speak(s,3,"NP is %d", np);
	xtg_speak(s,3,"xd is %9.2lf", xd);
	xtg_speak(s,3,"yd is %9.2lf", yd);
	xtg_speak(s,3,"zd is %9.2lf", zd);

	if (xd != UNDEF_POINT_IRAP) {
	    xtg_speak(s,3,"xd setting");
	    p_xp_v[j]=xd;
	    xtg_speak(s,3,"yd setting");
	    p_yp_v[j]=yd;
	    xtg_speak(s,3,"zd setting");
	    p_zp_v[j]=zd;
	    xtg_speak(s,3,"Values: %12.2lf, %12.2lf, %12,2lf",
		      p_xp_v[j],p_yp_v[j],p_zp_v[j]);
	}
	else{

	    xtg_speak(s,3,"Undef ...");
	    p_xp_v[j]=UNDEF_POINT;
	    p_yp_v[j]=UNDEF_POINT;
	    p_zp_v[j]=UNDEF_POINT;
	}

	xtg_speak(s,3,"Read line finished ...");
    }
    xtg_speak(s,1,"Imported: %d",np);
    xtg_speak(s,2,"Exit from <pox_import_rms>");
    fclose(fc);
    return np;
}
