/*
 * ############################################################################
 * pol_import_irap.c
 * Reads Irap Classic ASCII points format, dedicated to polygons
 * ############################################################################
 */


#include <string.h>
#include <stdlib.h>
#include "libxtg.h"
#include "libxtg_.h"

/*
 * ****************************************************************************
 *                        Irap points/polygon format
 * ****************************************************************************
 * 3.1001877e+03 1.6733066e+04 0.0000000e+00
 * 2.7623450e+03 1.7068479e+04 0.0000000e+00
 * 2.4968096e+03 1.7208236e+04 0.0000000e+00
 * ----------------------------------------------------------------------------
 */

void pol_import_irap (
		      int i1,
		      int i2,
		      double *p_xp_v,
		      double *p_yp_v,
		      double *p_zp_v,
		      char   *file,
		      int    debug
		      )
{
    FILE    *fc;
    int     iok, j, npoints;
    double  xd, yd, zd;
    char    s[24];

    strcpy(s,"pol_import_irap");
    xtgverbose(debug);

    /*
     *-------------------------------------------------------------------------
     * Open file and read
     *-------------------------------------------------------------------------
     */

    xtg_speak(s,2,"Entering <pox_import_irap>");
    xtg_speak(s,2,"Opening file: %s:", file);
    fc=fopen(file,"rb");
    if (fc==NULL) {
	xtg_error(s,"Cannot open file (NULL signal) %s", file);
    }
    xtg_speak(s,2,"Opening file ...DONE");

    npoints=999999999;

    /* tmp */
    for (j=1;j<=npoints;j++) {
      iok=fscanf(fc, "%lf %lf %lf", &xd, &yd, &zd);
      xtg_speak(s,3,"Read line: IOK is %d", iok);

      if (j >= i1 && j <= i2) {
	p_xp_v[j-i1]=xd;
	p_yp_v[j-i1]=yd;
	p_zp_v[j-i1]=zd;
      }
      if (j>i2) break;

    }

    xtg_speak(s,2,"Exit from import irap");
    fclose(fc);
    return;
}
