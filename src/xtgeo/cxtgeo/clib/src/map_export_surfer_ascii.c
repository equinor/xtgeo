/*
 * ############################################################################
 * map_export_surferasc.c
 * Export to SURFER ASCII Grid map format:
 *
 * DSAA
 * nx ny
 * xmin xmax
 * ymin ymax
 * zmin zmax (values outside will be defined as UNDEF)
 * ..... numbers .... X looping fastest ....
 *
 * 
 * Author: J.C. Rivenaes
 * Notice: A quick and dirty approach to zmin and zmax are done...
 *
 * ############################################################################
 */

#include "libxtg.h"
#include "libxtg_.h"

void map_export_surfer_ascii (
			      int nx, 
			      int ny, 
			      double xstep, 
			      double ystep,
			      double xmin, 
			      double xmax, 
			      double ymin, 
			      double ymax, 
			      double *p_map_v,
			      char  *file, 
			      int   debug
			      ) 
{
    
    int ib, i, j;
    float z;
    FILE *fc;
    char s[24]="map_export_surfer_a..";

    xtgverbose(debug);

    fc=fopen(file,"wb");
       
    xtg_speak(s,2,"Exporting to Surfer ASCII format");
    
    fprintf(fc,"DSAA\n");
    fprintf(fc,"%d %d\n%f %f\n%f %f\n-9000.0 9000.0\n",
	    nx, ny, xmin, xmax, ymin, ymax);
    
    for (j=1;j<=ny;j++) {
	for (i=1;i<=nx;i++) {
	    ib=x_ijk2ib(i,j,1,nx,ny,1,0);
	    z=p_map_v[ib];

	    if (z > UNDEF_MAP_LIMIT) {
		z=-9999.0;
	    }
	    fprintf(fc," %10.3f",z);
	}
	fprintf(fc,"\n");
    }
    fclose(fc);
}
