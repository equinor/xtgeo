/*
 * ############################################################################
 * pox_import_rms.c
 * Reads RMS internal points format (fpro points, polygons, ...
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
 *                        RMS points format (6.X)
 * ****************************************************************************
 * Begin GEOMATIC file header
 * type                                              = PolygonsComp
 * name                                              =
 * projectvisible                                    = FALSE
 * description                                       =
 * category                                          =
 * keyname                                           =
 * color                                             = 6.2745100e-01 1.2549020e-01
 * 9.4117647e-01
 * xpos                                              = -1
 * ypos                                              = -1
 * stamp                                             = 1013439487
 * ishorizontal                                      = FALSE
 * shrinkicons                                       = FALSE
 * xmin                                              = -7.7370747e+03
 * ymin                                              = -1.4702822e+04
 * zmin                                              = 0.0000000e+00
 * xmax                                              = 1.6392893e+04
 * ymax                                              = 1.7518129e+04
 * zmax                                              = 0.0000000e+00
 *
 * ....

 * datatype                                          = Polygons
 * points                                            = 76
 * polygons                                          = 3
 * filetype                                          = ASCII
 * fileversion                                       = 6.2000
 * End GEOMATIC file header
 * Begin parameter
 * id                                                = ParentParams
 * Begin parameter
 * id                                                = ComponentCoosysParams
 * fundamental                                       = ANY
 * x_shift                                           = -5.3275000000000000e+05
 * y_shift                                           = -6.7428500000000000e+06
 * z_shift                                           = -1.5394200439453125e+03
 * xy_scale                                          = 1.0000000000000000e+00
 * z_scale                                           = -1.0000000000000000e+00
 * xy_rotation                                       = 0.0000000000000000e+00
 * type                                              = XY
 * End parameter
 * End parameter
 * 3.1001877e+03 1.6733066e+04 0.0000000e+00   OR BINARY
 * 2.7623450e+03 1.7068479e+04 0.0000000e+00    "
 * 2.4968096e+03 1.7208236e+04 0.0000000e+00    "
 * ----------------------------------------------------------------------------
 * Comment:
 * Many of items in the header are not interesting
 * Todo:
 * Descent implementation of xshift, yshift and zshift
 * ----------------------------------------------------------------------------
 */

void pox_import_rms (
		     double *p_xp_v,
		     double *p_yp_v,
		     double *p_zp_v,
		     char   *file,
		     int    debug
		     )
{

    FILE    *fc;
    char    cline[133];
    int     i, j, npp, nep, npoints=0;
    char    key[50], eqsign[1], filetype[10];
    float   x, y, z;
    double  xd, yd, zd, xshift, yshift, zshift;
    char    s[24];

    strcpy(s,"pox_import_rms");
    xtgverbose(debug);

    /*
     *-------------------------------------------------------------------------
     * Open file
     *-------------------------------------------------------------------------
     */


    xtg_speak(s,2,"Entering <pox_import_rms>");
    xtg_speak(s,2,"Opening file: ...");
    fc=fopen(file,"r");
    if (fc==NULL) {
	xtg_error(s,"Cannot open file");
    }
    xtg_speak(s,2,"Opening file ...DONE");

    /*
     *-------------------------------------------------------------------------
     * Read header and look for usable keywords:
     *-------------------------------------------------------------------------
     */
    nep=0;  /* number of "End parameter" lines */
    for (i=1;i<1000;i++) {
	fgets(cline,132,fc);
	/* look for <points> */
	if (strncmp(cline,"points", 6) == 0) {
	    xtg_speak(s,2,"Keyword <points> found");
	    sscanf(cline,"%s %s %d", key, eqsign, &npp);
	    xtg_speak(s,2,"Number of points is %d", npp);
	    npoints=npp;
	}
	if (strncmp(cline,"polygons", 8) == 0) {
	    xtg_speak(s,2,"Keyword <polygons> found");
	    sscanf(cline,"%s %s %d", key, eqsign, &npp);
	    xtg_speak(s,2,"Number of polygons is %d", npp);
	    npoints=npoints+npp-1; /*each polygon ends with -999.0000 line flagh except last */
	}
	if (strncmp(cline,"x_shift", 7) == 0) {
	    xtg_speak(s,2,"Keyword <x_shift> found");
	    sscanf(cline,"%s %s %lf", key, eqsign, &xshift);
	}
	if (strncmp(cline,"y_shift", 7) == 0) {
	    xtg_speak(s,2,"Keyword <y_shift> found");
	    sscanf(cline,"%s %s %lf", key, eqsign, &yshift);
	}
	if (strncmp(cline,"z_shift", 7) == 0) {
	    xtg_speak(s,2,"Keyword <z_shift> found");
	    sscanf(cline,"%s %s %lf", key, eqsign, &zshift);
	}
	if (strncmp(cline,"filetype", 8) == 0) {
	    xtg_speak(s,2,"Keyword filetype found");
	    sscanf(cline,"%s %s %s", key, eqsign, filetype);
	    xtg_speak(s,2,"Filetype is %s", filetype);
	}
	if (strncmp(cline,"End parameter", 13) == 0) {
	    xtg_speak(s,2,"Keyword <End parameter> found");
	    nep+=1;
	}
	if (nep == 2) {
	    for (j=0;j<npoints;j++) {
		if (strncmp(filetype,"ASCII",5)==0) {
		    fscanf(fc, "%f %f %f", &x, &y, &z);
		}
		else{
		    fread(&x,4,1,fc);
		    if (x_swap_check()) SWAP_FLOAT(x);
		    fread(&y,4,1,fc);
		    if (x_swap_check()) SWAP_FLOAT(y);
		    fread(&z,4,1,fc);
		    if (x_swap_check()) SWAP_FLOAT(z);
		}
		xd=x;
		yd=y;
		zd=z;
		if (x != UNDEF_POINT_RMS) {
		    p_xp_v[j]=xd-xshift;
		    p_yp_v[j]=yd-yshift;
		    p_zp_v[j]=-1*(zd)-zshift;
		}
		else{
		    p_xp_v[j]=UNDEF_POINT;
		    p_yp_v[j]=UNDEF_POINT;
		    p_zp_v[j]=UNDEF_POINT;
		}
	    }
	    xtg_speak(s,2,"Exit from <pox_import_rms>");
	    return;
	}

    }
}
