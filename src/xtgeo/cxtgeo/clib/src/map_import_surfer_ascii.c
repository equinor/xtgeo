/*
 * ############################################################################
 * map_import_surfer_ascii.c
 * Import Surfer format. See export of Surfer for format spec.
 * Author: J.C. Rivenaes
 * ############################################################################
 */

#include "libxtg.h"
#include "libxtg_.h"

void map_import_surfer_ascii (
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
    int i, nxy, iok=0;
    float value;
    FILE *fd;
    char sub[24], string[8];

    xtgverbose(debug);

    strcpy(sub,"map_import_surfer_as..");

    /* read header */
    xtg_speak(sub,2,"Entering routine");


    fd=fopen(file,"r");
    if (fd == NULL) {
	xtg_speak(sub,2,"Opening Surfer file!");
    }
    else{
	xtg_speak(sub,2,"Opening Surfer file ...OK!");
    }

    /*initial settings*/
    *numdef     = 0;
    *numdefsum  = 0;
    *zmin       = 1.0E31;
    *zmax       = -1.0E31;


    /* read header */
    xtg_speak(sub,2,"Reading header!");
    fscanf(fd,"%s %d %d %lf %lf %lf %lf %lf %lf",
	   string, nx, ny, xmin, xmax, ymin, ymax, zmin, zmax);

    *xstep = (*xmax - *xmin)/(*nx-1);
    *ystep = (*ymax - *ymin)/(*ny-1);


    /* read values */

    nxy= *ny * *nx;
    xtg_speak(sub,2,"Rows and columns: %d x %d", *nx, *ny);

    for (i=0; i<nxy; i++) {
	iok=fscanf(fd,"%f",&value);
	if (value < *zmin || value > *zmax) {
	    value=UNDEF_MAP;
	    p_map_v[i]=value;
	}
	else{
	    *numdef = *numdef + 1;
	    *numdefsum = *numdefsum + i;
	    p_map_v[i]=value;
	}
    }

    xtg_speak(sub,2,"Reading data OK %d", iok);
    if (iok != 0) *ierr = iok;

    *ierr=0;

}
