/*
 * ############################################################################
 * grd3d_import_grdeclpar.c
 * Reading an Eclipse ASCII parameter
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
 * This routine is based on that every thing regarding the GRID is known
 * (geometry, nx, ny, nz, ...)
 * ----------------------------------------------------------------------------
 * It looks for one parameter in the file and imports that, assuming it to be
 * double (it may be converted later)
 *
 */


void grd3d_import_grdeclpar (
			     int      nx,
			     int      ny,
			     int      nz,
			     char     *param_name,
			     double   *p_dfloat_v,
			     char     *filename,
			     int      debug
			     )


{
    char   cline[9];
    FILE   *fc;
    int    nlen, num, ier=0, line, i;
    float  myfloat;
    char   s[24]="grd3d_import_grdeclpar";


    xtgverbose(debug);
    xtg_speak(s,2,"Entering routine ...");


    /*
     *-------------------------------------------------------------------------
     * Open file
     *-------------------------------------------------------------------------
     */

    xtg_speak(s,2,"Opening GRDECL file...");
    fc=fopen(filename,"r");
    if (fc==NULL) {
	xtg_error(s,"Cannot open file!");
    }
    xtg_speak(s,2,"Opening GRDECL file...OK!");



    /*
     *=========================================================================
     * Loop file... It is NOT necessary to do many tests; that should be done
     * by the calling PERL script? The Perl routine will also secure that
     * the requested parameter really exist...
     *=========================================================================
     */

    nlen=strlen(param_name);
    xtg_speak(s,2,"Length of string %s is %d",param_name,nlen);
    num=nx*ny*nz;

    xtg_speak(s,3,"Scanning file %s for param %s ...", filename, param_name);
    for (line=1;line<999999999;line++) {

	/* Read keywords */
	if (fgets(cline, 8, fc) != NULL) xtg_speak(s, 4, "CLINE \n%s", cline);

	/*
	 *---------------------------------------------------------------------
	 * Getting 'parameter' values
	 *---------------------------------------------------------------------
	 */
	if (strncmp(cline, param_name, nlen) == 0) {
	    xtg_speak(s,3,"Parameter was found: %s", param_name);
	    for (i=0; i<num;  i++) {
		ier=fscanf(fc,"%f",&myfloat);
		p_dfloat_v[i]=myfloat;
		if (ier != 1) {
		    xtg_error(s,"Error during read of float! Check file!");
		}
	    }

	    xtg_speak(s,2,"Reading parameter: %s ... DONE!", param_name);
	    break;

	}
    }

    fclose(fc);


    xtg_speak(s,2,"Leaving routine ...");
}
