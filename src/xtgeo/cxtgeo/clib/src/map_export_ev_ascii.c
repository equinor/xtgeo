/*
 * ############################################################################
 * map_export_ev_ascii.c
 * Export to EARTHVISION ASCII map format:
 * ----------------------------------
 * # Type: scattered data
 * # Version: 6
 * # Description: Exported from RMS - Reservoir Modelling System, version 2013.1.3
 * # Format: free
 * # Field: 1 x
 * # Field: 2 y
 * # Field: 3 z meters
 * # Field: 4 column integer
 * # Field: 5 row integer
 * # Projection: Local Rectangular
 * # Units: meters
 * # End:
 * # Information from grid:
 * # Grid_size: 481 x 521
 * # Grid_space: 451000.000000,463000.000000,6779000.000000,6792000.000000
 * # Z_field: z
 * # History: No history
 * # Z_units: meters
 * 451000.0000000 6779000.0000000 0.0000000 1 1
 * 451025.0000000 6779000.0000000 0.0000000 2 1
 * ...
 * ..... numbers .... X looping fastest ....
 *
 * The output is for unrotated maps
 *
 * Author: J.C. Rivenaes
 *
 * ############################################################################
 */

#include "libxtg.h"
#include "libxtg_.h"

void map_export_ev_ascii (
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
    double z, xs, ys;
    FILE *fc;
    char s[24]="map_export_ev_ascii";

    xtgverbose(debug);

    fc=fopen(file,"wb");

    xtg_speak(s,2,"Exporting to Surfer ASCII format");

    fprintf(fc,
            "# Type: scattered data\n"
            "# Version: 6\n"
            "# Description: Exported from XTGeo"
            "# Format: free\n"
            "# Field: 1 x\n"
            "# Field: 2 y\n"
            "# Field: 3 z meters\n"
            "# Field: 4 column integer\n"
            "# Field: 5 row integer\n"
            "# Projection: Local Rectangular\n"
            "# Units: meters\n"
            "# End:\n"
            "# Information from grid:\n"
            "# Grid_size: %d x %d\n"
            "# Grid_space: %10.2f %10.2f %10.2f %10.2f\n"
            "# Z_field: z\n"
            "# History: No history\n"
            "# Z_units: meters\n",
            nx, ny, xmin, xmax, ymin, ymax);


    for (j=1;j<=ny;j++) {
	for (i=1;i<=nx;i++) {
	    xs = xmin + xstep*(i-1);
	    ys = ymin + ystep*(j-1);

	    ib=x_ijk2ib(i,j,1,nx,ny,1,0);
	    z=p_map_v[ib];

	    if (z > UNDEF_MAP_LIMIT) {
		z=-9999.0;
	    }
	    fprintf(fc,"%10.2f  %10.2f  %11.5f  %d  %d\n",xs, ys, z, i, j);
	}
    }
    fclose(fc);
}
