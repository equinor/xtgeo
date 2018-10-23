/*
 * #################################################################################################
 * grd3d_list_ecl_kw.c
 * Basic routine that lists ECLIPSE keywords in INIT/GRID/RESTART, and also display
 * RESTART dates
 * #################################################################################################
 */


#include <string.h>
#include <stdlib.h>
#include "libxtg.h"
#include "libxtg_.h"

/*
 * *************************************************************************************************
 * Formatting; see grd3d_import_ecl_props.c
 * -------------------------------------------------------------------------------------------------
 * Parameter list:
 * ftype                Type of file:
 *                      1 = Binary INIT file
 *                      2 = ASCII  INIT file
 *                      3 = Binary RESTART file (non-unified)
 *                      4 = ASCII  RESTART file (non-unified)
 *                      5 = Binary RESTART file (unified)
 *                      6 = ASCII  RESTART file (unified)
 *                      7 = Binary GRID or EGRID file
 *                      8 = ASCII  GRID/EGRID file
 * filename             Name of file to read from
 * debug                Debug (verbose) level
 * -------------------------------------------------------------------------------------------------
 */

void grd3d_list_ecl_kw (
			int    ftype,
			char   *filename,
			int    debug
			)
{


    int     i, ios = 0, nmax;
    char    cname[9], ctype[5], fmode[3];
    char    s[24]="grd3d_list_ecl_kw";

    int     nact, nsec = 0, aday, amonth, ayear;
    int     dtrigger=0, dflag=0;

    int     *tmp_int_v;
    float   *tmp_float_v;
    double  *tmp_double_v;
    int     *tmp_logi_v;
    char    **tmp_string_v;

    FILE    *fc;


    xtgverbose(debug);


    /*
     *----------------------------------------------------------------------------------------------
     * Initialize...
     * I now need to allocate space for tmp_* arrays
     *----------------------------------------------------------------------------------------------
     */

    nmax=100000000;

    tmp_int_v    = calloc(nmax, sizeof(int));
    tmp_float_v  = calloc(nmax, sizeof(float));
    tmp_double_v = calloc(nmax, sizeof(double));
    tmp_logi_v   = calloc(nmax, sizeof(int));


    /* the string vector is 2D */
    tmp_string_v=calloc(nmax, sizeof(char *));
    for (i=0; i<nmax; i++) tmp_string_v[i]=calloc(9, sizeof(char));

    /* trig date output for RESTART */
    if (ftype >=3 && ftype <=6) {
	dtrigger=1;
    }


    /*
     *----------------------------------------------------------------------------------------------
     * Open file and more initial work
     *----------------------------------------------------------------------------------------------
     */
    xtg_speak(s,2,"Opening %s",filename);

    strcpy(fmode,"r");
    if (ftype==1 || ftype==3 || ftype==5) strcpy(fmode,"rb");

    fc=fopen(filename,fmode);

    xtg_speak(s,2,"Finish opening %s",filename);


    /*
     *==============================================================================================
     * START READING!
     *==============================================================================================
     */


    while (ios == 0) {

	if (ftype == 1 || ftype==3 || ftype==5 || ftype==7) {
	    ios=u_read_ecl_bin_record (
				       cname,
				       ctype,
				       &nact,
				       nmax,
				       nmax,
				       nmax,
				       nmax,
				       nmax,
				       tmp_int_v,
				       tmp_float_v,
				       tmp_double_v,
				       tmp_string_v,
				       tmp_logi_v,
				       fc,
				       debug
				       );
	}
	else{
	    /* the ASCII reader is not optimal (trouble with CHAR records) */
	    ios=u_read_ecl_asc_record (
				       cname,
				       ctype,
				       &nact,
				       tmp_int_v,
				       tmp_float_v,
				       tmp_double_v,
				       tmp_string_v,
				       tmp_logi_v,
				       fc,
				       debug
				       );
	}


	if (ios != 0) break;


 	/* a SEQNUM will always come before the INTEHEAD in restart files */
	if (dtrigger == 1 && strncmp(cname, "SEQNUM  ",8) == 0) {
	    xtg_speak(s,2,"New SEQNUM record");
	    nsec   = tmp_int_v[0];
	    dflag=1;
	}


	if (dflag==1 && strncmp(cname, "INTEHEAD",8) == 0) {
	    aday   = tmp_int_v[64];
	    amonth = tmp_int_v[65];
	    ayear  = tmp_int_v[66];

	    xtg_speak(s,2,"INTEHEAD found...");
	    xtg_speak(s,2,"Date found is %d %d %d ...",aday, amonth, ayear);
	    dflag=2;

	}


	if (dflag==2) {
	    xtg_shout(s,"%-s   %9d  %5s  >>> (SEQ: %d)  %4d %02d %02d",cname, nact,ctype, nsec,
		      ayear, amonth, aday);
	    dflag=0;
	}
	else{
	    xtg_shout(s,"%-s   %9d  %5s",cname, nact,ctype);
	}
    }

}
