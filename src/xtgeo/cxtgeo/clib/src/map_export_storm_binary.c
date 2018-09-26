/*
 * ############################################################################
 * map_export_storm_binary.c
 * Export to Storm binary map format
 * Author: J.C. Rivenaes
 * ############################################################################
 */

#include "libxtg.h"
#include "libxtg_.h"

/*
 * ****************************************************************************
 *                        GRD2D_EXPORT_STORM_BINARY
 * ****************************************************************************
 * The format is rather simple; regular XY with no rotation
 * The field values are double precision
 */
void map_export_storm_binary (
				int    nx, 
				int    ny, 
				double xstep, 
				double ystep,
				double xmin, 
				double xmax, 
				double ymin, 
				double ymax, 
				double *p_map_v,
				char   *file, 
				int    debug
				) 
{
    
    int i, nxy;
    double dbl_value;
    FILE *fc;
    int swap;
    char s[24]="map_export_storm_bi..";

    xtgverbose(debug);
    swap=x_swap_check();

    xtg_speak(s,1,"Remove existing file and open new");
    fc=fopen(file,"w");

       
    xtg_speak(s,1,"Exporting to Storm binary format");
    
    fprintf(fc,"STORMGRID_BINARY\n\n");
    fprintf(fc,"%d %d %f %f\n%f %f %f %f\n",
	    nx, ny, xstep, ystep, xmin, xmax, ymin, ymax);

    nxy = nx * ny;
    
    for (i=0;i<nxy;i++) {
	dbl_value=p_map_v[i];

	if (dbl_value > UNDEF_MAP_LIMIT) {
	    dbl_value=UNDEF_MAP_STORM;
	}

	/* byte swapping if needed */
	if (swap==1) SWAP_DOUBLE(dbl_value);

	fwrite(&dbl_value,8,1,fc);
    }

    fclose(fc);

}
