/*
 * ############################################################################
 * map_import_irap_ascii.c
 * Import Irap Classic map on ascii format
 * Author: J.C. Rivenaes
 * ############################################################################
 * $Id: map_import_irap_ascii.c,v 1.1 2001/03/14 08:02:29 bg54276 Exp bg54276 $
 * $Source: /d/proj/bg/avresmod/src/gplib/GPLExt/RCS/map_import_irap_ascii.c,v $
 *
 * $Log: map_import_irap_ascii.c,v $
 * Revision 1.1  2001/03/14 08:02:29  bg54276
 * Initial revision
 *
 * Revision 1.1  2000/12/12 17:24:54  bg54276
 * Initial revision
 *
 *
 * ############################################################################
 */

#include "libxtg.h"
#include "libxtg_.h"

/*
 * ****************************************************************************
 *                        GRD2D_IMPORT_IRAP_ASCII
 * The values of the maps are treated as a single dimension array p_map_v
 * ****************************************************************************
 */

void map_import_irap_ascii (
			      int   *numdef,
			      int   *numdefsum,
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
			      double *p_map_v,
			      char  *file,
			      int   *ierr,
			      int   debug
			      )
{

    /* locals*/
    int    idum, i, nxy, iok=0;
    FILE   *fd;
    char   sub[24] ;

    float fxmin, fxmax, fxstep, fymin, fymax, fystep, rdum, value;

    xtgverbose(debug);

    strcpy(sub,"map_import_irap_ascii");


    /* read header */
    xtg_speak(sub,2,"Entering routine");


    fd=fopen(file,"r");
    if (fd == NULL) {
	xtg_speak(sub,2,"Opening Irap FGR file FAILED!");
    }
    else{
	xtg_speak(sub,2,"Opening Irap FGR file...OK!");
    }

    /*initial settings*/
    *numdef    = 0;
    *numdefsum = 0;
    *zmin      = 1.0E31;
    *zmax      = -1.0E31;


    /* read header */
    xtg_speak(sub,2,"Reading header!");
    fscanf(fd,"%d %d %f %f %f %f %f %f %d %f %f %f %d %d %d %d %d %d %d",
	   &idum, ny, &fxstep, &fystep,
	   &fxmin, &fxmax, &fymin, &fymax,
	   nx, &rdum, &rdum, &rdum,
	   &idum, &idum, &idum, &idum, &idum, &idum, &idum);

    *xmin  = fxmin;
    *xmax  = fxmax;
    *xstep = fxstep;

    *ymin  = fymin;
    *ymax  = fymax;
    *ystep = fystep;


    /* read values */

    nxy= *ny * *nx;
    xtg_speak(sub,2,"Rows and columns: %d x %d", *nx, *ny);

    for (i=0; i<nxy; i++) {
	iok=fscanf(fd,"%f",&value);
	if (value == UNDEF_MAP_IRAP) {
	    value=UNDEF_MAP;
	    p_map_v[i]=value;
	}
	else{
	    *numdef = *numdef + 1;
	    if (value < *zmin) *zmin=value;
	    if (value > *zmax) *zmax=value;
	    p_map_v[i]=value;
	    *numdefsum = *numdefsum + i; /* a sort of unique identifier */
	}
    }

    xtg_speak(sub,2,"Reading data OK %d", iok);
    if (iok != 0) *ierr = iok;

    *ierr=0;

}
