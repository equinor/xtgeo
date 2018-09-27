/*
 * ############################################################################
 * grd3d_scan_ecl_egrid_hd.c
 * Scan records of an Eclipse EGRID file in order to return NX, NY, NZ
 * Author: JRIV
 * ############################################################################
 */


#include <string.h>
#include <stdlib.h>
#include "libxtg.h"
#include "libxtg_.h"

/*
 * ****************************************************************************
 *                        GRD3D_IMPORT_ECLIPSE_GRID
 * ****************************************************************************
 * The format is the Eclipse binary/ascii output; the EGRID or FEGRID file,
 * This is a corner-point format.
 *
 * The EGRID file is aka a binary version of GRDECL file with some headers. For
 * dimensions, the GRIDHEAD is the one to use:
 * 'GRIDHEAD'         100 'INTE'
 *         1          20          20           2           0           0
 *         0           0           0           0           0           0
 * etc
 * Read item 2,3,4 as NX NY NZ, here 20 x 20 x 2
 * For the binary form, the record starts and ends with a 4 byte integer, that
 * says how long the current record is, in bytes.
 *
 * ----------------------------------------------------------------------------
 *
 */


void grd3d_scan_ecl_egrid_hd (
			     int   mode,
			     int   *nx,
			     int   *ny,
			     int   *nz,
			     char  *filename,
			     int   debug
			     )
{

    /* locals */
    int            ios=0, reclen;
    int            i, doread, item;
    float          *tmp_float_v;
    double         *tmp_double_v;
    int            *tmp_int_v, *tmp_logi_v;
    char           **tmp_string_v;
    int            max_alloc_int, max_alloc_float, max_alloc_double;
    int   	   max_alloc_char, max_alloc_logi;

    /* length of char must include \0 termination? */
    char           cname[9], ctype[5];
    char           s[24]="grd3d_scan.._egrid_hd";

    FILE *fc;

    /*
     * ========================================================================
     * INITIAL TASKS
     * ========================================================================
     */

    xtgverbose(debug);
    xtg_speak(s,2,"Entering routine");

    fc=fopen(filename,"rb");
    xtg_speak(s,2,"Finish opening %s",filename);

    max_alloc_int    = 200000;
    max_alloc_float  = 200000;
    max_alloc_double = 200000;
    max_alloc_logi   = 200000;
    max_alloc_char   = 200000;


    tmp_int_v    = calloc(max_alloc_int, sizeof(int));
    tmp_float_v  = calloc(max_alloc_float, sizeof(float));
    tmp_double_v = calloc(max_alloc_double, sizeof(double));
    tmp_string_v = calloc(max_alloc_char, sizeof(char *));
    for (i=0; i<1; i++) tmp_string_v[i]=calloc(9, sizeof(char));
    tmp_logi_v   = calloc(max_alloc_logi, sizeof(int));

    /*
     * ========================================================================
     * READ RECORDS AND COLLECT NECESSARY STUFF
     * ========================================================================
     */

    doread=1;
    item=0;

    while (doread==1) {
	item++;

	ios=u_read_ecl_bin_record (
				   cname,
				   ctype,
				   &reclen,
				   max_alloc_int,
				   max_alloc_float,
				   max_alloc_double,
				   max_alloc_char,
				   max_alloc_logi,
				   tmp_int_v,
				   tmp_float_v,
				   tmp_double_v,
				   tmp_string_v,
				   tmp_logi_v,
				   fc,
				   debug
				   );


	xtg_speak(s,2,"Reading item no %d: %s",item,cname);

	if (strcmp(cname,"GRIDHEAD")==0) {
	    xtg_speak(s,2,"Reading GRIDHEAD values...");
	    *nx=tmp_int_v[1];
	    *ny=tmp_int_v[2];
	    *nz=tmp_int_v[3];
	    xtg_speak(s,2,"Found DIMENS %d x %d x %d = %d",
		      *nx,
		      *ny,
		      *nz,
		      *nx * *ny * *nz);

	doread=0;
	}

	if (item>9) doread=0;

    }
    /* free allocated space */
    xtg_speak(s,2,"Freeing allocated space...");

    free(tmp_int_v);
    free(tmp_float_v);
    free(tmp_double_v);

    /* I am getting a bus error on this sometimes... */
    /*
    for (i=0; i<1; i++) free(tmp_string_v[i]);
    free(tmp_string_v);
    */
    free(tmp_logi_v);
    xtg_speak(s,2,"Freeing allocated space... DONE");

}
