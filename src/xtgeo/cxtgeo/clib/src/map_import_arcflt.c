/*
 * ############################################################################
 * map_import_arcflt.c
 * Import ARCINFO FLOAT binary map format (unrotated map, regular format)
 * Author: J.C. Rivenaes
 * UNFINISHED: CHECK NEEDED! BYTEODER NOT COVERED FULLY!
 * ############################################################################
 */


#include "libxtg.h"
#include "libxtg_.h"


/*
 * ****************************************************************************
 *                        ARCINFO FLOAT FORMAT
 *             For format, see corresponding export routine
 * ****************************************************************************
 */

void map_import_arcflt (
			int   *ndef,
			int   *ndefsum,
			int   *nx,
			int   *ny,
			double *xstep,
			double *ystep,
			double *xmin,
			double *xmax,
			double *ymin,
			double *ymax,
			double *zmin,
			double *zmax,
			double *map_v,
			char  *file1,
			char  *file2,
			int   *ierr,
			int   debug
			)
{

    /* locals */
    int i, j, ib, nxx, nyy, iok_close, not_def, idum;
    float myvalue, undefv;
    char string[30], byteorder[9];
    FILE *fc;
    char sub[24]="map_import_arcflt";
    int  swap;

    idum=xtgverbose(debug);

    xtg_speak(sub,4,"Setting VERBOSE....");

    swap=x_swap_check();

    /* ierr=0 if all OK */
    *ierr=0;

    /* The Perl class should do a check if file exist! */
    xtg_speak(sub,2,"Opening file %s",file1);
    fc=fopen(file1,"rb");
    xtg_speak(sub,2,"Reading ASCII header file ...");

    /* header is ASCII. NB remember the \n !... */
    xtg_speak(sub,2,"Scanning header...");
    fscanf(fc,"%s %d\n",string, nx);           xtg_speak(sub,2,"HALLO NX is %d", *nx);
    fscanf(fc,"%s %d\n",string, ny);           xtg_speak(sub,2,"NY is %d", *ny);
    fscanf(fc,"%s %lf\n",string, xmin);         xtg_speak(sub,2,"XMIN is %f", *xmin);
    fscanf(fc,"%s %lf\n",string, ymin);         xtg_speak(sub,2,"YMIN is %f", *ymin);
    fscanf(fc,"%s %lf\n",string, xstep);        xtg_speak(sub,2,"XSTEP is %f", *xstep);
    fscanf(fc,"%s %f\n",string, &undefv);       xtg_speak(sub,2,"UNDEFV is %f", undefv);
    fscanf(fc,"%s %s\n",string, byteorder);    xtg_speak(sub,2,"BYTEORDER is %s", byteorder);
    xtg_speak(sub,2,"Scanning header...OK");
    *ystep = *xstep;
    fclose(fc);

    nxx=*nx;
    nyy=*ny;

    xtg_speak(sub,2,"NX and NY is: %d %d",*nx, *ny);
    not_def=0;
    *ndefsum=0;

    fc=fopen(file2,"rb");

    for (j=nyy; j>=1; j--) {
	for (i=1; i<=nxx; i++) {
	    ib=x_ijk2ib(i,j,1,nxx,nyy,1,0);

	    /* read a single value at the time */

	    fread (&myvalue,4,1,fc);

	    /* if (swap==1) SWAP_DOUBLE(dbl_value);*/

	    if (myvalue == undefv) {
		not_def++;

		/* return error code if strange map value */
		if (myvalue > UNDEF_MAP) {
		    *ierr=2414;
		}
		myvalue = UNDEF_MAP;
	    }
	    else{
		*ndefsum = *ndefsum + ib;
	    }

	    /* find minimum and maximum map value */
	    if (myvalue < UNDEF_MAP_LIMIT && myvalue < *zmin) *zmin=myvalue;
	    if (myvalue < UNDEF_MAP_LIMIT && myvalue > *zmax) *zmax=myvalue;

	    map_v[ib]=myvalue;
	}
    }

    *ndef= *nx * *ny - not_def;
    *xmax = *xmin  + *nx * *xstep;
    *ymax = *ymin + *ny * *ystep;

    xtg_speak(sub,2,"ZMIN and ZMAX: %f   %f\n", zmin, zmax);

    iok_close=fclose(fc);

    if (iok_close != 0) *ierr=iok_close;

}
