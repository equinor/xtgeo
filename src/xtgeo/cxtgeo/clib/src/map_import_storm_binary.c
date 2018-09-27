/*
 * ############################################################################
 * map_import_storm_binary.c
 * Import Storm binary map format (unrotated map, regular format)
 * Author: J.C. Rivenaes
 * ############################################################################
 * $Id: $ 
 * $Source: $ 
 *
 * $Log: map_import_storm_binary.c,v $
 *
 *
 * ############################################################################
 */


#include "libxtg.h"
#include "libxtg_.h"


/*
 * ****************************************************************************
 *                        GRD2D_IMPORT_STORM_BINARY
 * ****************************************************************************
 * The format is rather simple; regular XY with no rotation
 */

void map_import_storm_binary (
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
				char  *file, 
				int   *ierr,
				int   debug
				) 
{

    /* locals */
    int     i, nxy, iok_close, not_def, idum;
    double  dbl_value;
    double  myvalue;
    char    string[30];
    FILE    *fc;
    char    sub[24]="map_import_storm_bi..";
    int     swap;
    double  fxstep, fystep, fxmin, fxmax, fymin, fymax; 


    idum=xtgverbose(debug);

    xtg_speak(sub,4,"Setting VERBOSE....");

    swap=x_swap_check();

    /* ierr=0 if all OK */
    *ierr=0;

    /* The Perl class should do a check if file exist! */
    xtg_speak(sub,2,"Opening file %s",file);
    fc=fopen(file,"rb");
    xtg_speak(sub,2,"Reading map file %s",file);
       
    /* header is ASCII. NB remember the \n !... */
    xtg_speak(sub,2,"Scanning header...");
    fscanf(fc,"%s %d %d %lf %lf %lf %lf %lf %lf\n",
	   string, nx, ny, &fxstep, &fystep, &fxmin, &fxmax, &fymin, &fymax);
    xtg_speak(sub,2,"Scanning header...OK");
    xtg_speak(sub,3,"NX NY XSTEP YSTEP XMIN XMAX YMIN YMAX %d %d %10.2f %10.2f %10.2f %10.2f %10.2f %10.2f", 
	      *nx, *ny, fxstep, fystep, fxmin, fxmax, fymin, fymax);
    

    *xstep = fxstep;
    *ystep = fystep;
    *xmin  = fxmin;
    *ymin  = fymin;
    *xmax  = fxmax;
    *ymax  = fymax;


    *zmin=99999.9;
    *zmax=-99999.9;

    nxy=*nx * *ny;
    xtg_speak(sub,2,"NX and NY is: %d %d of total %d",*nx, *ny, nxy);
    not_def=0;
    *ndefsum=0;


    for (i=0; i<nxy; i++) {

	/* read a single value at the time */
	
	fread (&dbl_value,8,1,fc);

	if (swap==1) SWAP_DOUBLE(dbl_value);

	/* implicit convert to double */
	myvalue=dbl_value;
	/*xtg_speak(sub,4,"Reading value no %d as %f",i,myvalue);*/

	if (myvalue == UNDEF_MAP_STORM) {
	    not_def++;

	    /* return error code if strange map value */
	    if (myvalue > UNDEF_MAP) {
		*ierr=2414;
	    }
	    myvalue = UNDEF_MAP;
	}
	else{
	    *ndefsum = *ndefsum + i;
	}

	/* find minimum and maximum map value */
	if (myvalue < UNDEF_MAP_LIMIT && myvalue < *zmin) *zmin=myvalue;
	if (myvalue < UNDEF_MAP_LIMIT && myvalue > *zmax) *zmax=myvalue;
	
	map_v[i]=myvalue;
    }

    *ndef= nxy - not_def;

    iok_close=fclose(fc);

    if (iok_close != 0) *ierr=iok_close;

}

