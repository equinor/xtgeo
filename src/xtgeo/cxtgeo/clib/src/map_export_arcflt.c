/*
 * ############################################################################
 * map_export_arcflt.c
 * Export to ARCINFO binary map format
 * Author: J.C. Rivenaes
 * ############################################################################
 */

#include "libxtg.h"
#include "libxtg_.h"

/*
 * ****************************************************************************
 *                         ARCINFO
 * ****************************************************************************
 * The format has two files, one ASCII header file and one binary file
 * The ASCII header file example (*.hdr):
 * ncols 288
 * nrows 133
 * cellsize 50
 * xllcorner 149239.0
 * yllcorner 6702030.0
 * byteorder lsbfirst         # or msbfirst for big-endian
 * nodata_value 99999.0       # for example
 */
void map_export_arcflt (
			int nx, 
			int ny, 
			double xstep, 
			double ystep,
			double xmin, 
			double xmax, 
			double ymin, 
			double ymax, 
			double *p_map_v,
			char  *file1, 
			char  *file2, 
			int   debug
			) 
{
    
    int i, j, ib;
    double dbl_value;
    float val;
    FILE *fc;
    int swap;
    char s[24]="map_export_arcflt";

    xtg_speak(s,2,"Enter routine ...");

    xtgverbose(debug);
    swap=x_swap_check();

    fc=fopen(file1,"wb");
       
    xtg_speak(s,2,"Exporting to ARCFLT header format");
    xtg_speak(s,2,"Header file: %s", file1);
    
    xtg_speak(s,2,"XMIN etc %f %f %f", xmin, ymin, xstep);


    fprintf(fc,"ncols %d\n",nx);
    fprintf(fc,"nrows %d\n",ny);    
    fprintf(fc,"xllcorner %f\n",xmin);
    fprintf(fc,"yllcorner %f\n",ymin);
    fprintf(fc,"cellsize %f\n",xstep);
    /* reuse Storm default as UNDEF value */
    fprintf(fc,"nodata_value %f\n",UNDEF_MAP_STORM);
    if (swap==1) {
	fprintf(fc,"byteorder lsbfirst\n");
    }
    else{
	fprintf(fc,"byteorder msbfirst\n");
    }
    
    fclose(fc);
    
    /*
     * Now write the binary data to a file
     */
    
    fc=fopen(file2,"wb");
    
    
    for (j=ny;j>=1;j--) {
	for (i=1;i<=nx;i++) {
	    ib=x_ijk2ib(i,j,1,nx,ny,1,0);

	    dbl_value=p_map_v[ib];
	    if (dbl_value > UNDEF_MAP_LIMIT) {
		dbl_value=UNDEF_MAP_STORM;
	    }
	    val=dbl_value;
	    fwrite(&val,4,1,fc);
	}
    }

    fclose(fc);
}
