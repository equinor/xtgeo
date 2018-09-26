/*
 * ############################################################################
 * u_read_ecl_asc_record.c
 * Reading an Eclipse ASCII record
 * Author: JCR
 * ############################################################################
 * $Id:  $ 
 * $Source: $ 
 *
 * $Log: $
 *
 * ############################################################################
 */


#include <string.h>
#include <stdlib.h>
#include "libxtg.h"
#include "libxtg_.h"


    
/*
 * ****************************************************************************
 *                      READ_ECLIPSE_ASCII_RECORD
 * ****************************************************************************
 * The format is the Eclipse binary/ascii output; the GRID/INIT etc file, 
 * Records may look like:
 *
 * 'COORDS  '           7 'INTE'
 *           8           1           1           8           0           0
 *           0
 * 'CORNERS '          24 'REAL'
 *    .41300209E+06    .71332070E+07    .35028701E+04    .41314041E+06
 *    .71331340E+07    .35261799E+04    .41294591E+06    .71331490E+07
 *    .34913301E+04    .41307969E+06    .71330730E+07    .35180801E+04
 *    .41300209E+06    .71332070E+07    .35029500E+04    .41314041E+06
 *    .71331340E+07    .35263501E+04    .41294591E+06    .71331490E+07
 *    .34914500E+04    .41307969E+06    .71330730E+07    .35182900E+04
 * etc
 * 
 * Notice: 1. The '' are from Fortran, and must be treated carefully here.
 *         2. Many C compilers cannot take the Fortran D (double) format.
 * ----------------------------------------------------------------------------
 *
 */   


int u_read_ecl_asc_record (
			     char    *cname,
			     char    *ctype,
			     int     *reclen,
			     int     *tmp_int_v,
			     float   *tmp_float_v,
			     double  *tmp_double_v,
			     char    **tmp_string_v,
			     int     *tmp_logi_v,
			     FILE    *fc,
			     int     debug
			     )

/*
 * This routine reads the Eclipse standard format
 */

{

    int     ivalue;
    float   fvalue;
    double  dvalue;
    char    cvalue;
    char    sdvalue[30]; /* applied when reading DOUBLE PRECISION D format */

    int   iok;
    int   i, irec, j, jj, jline, m;
    char  cline[100];

    char  cv[8][9];
    char  s[24]="u_read_ecl_asc_record";

    xtgverbose(debug);

    /* read the description line, as e.g.:
       'CORNERS '          24 'REAL'<16>
    */

    if (fgets(cline,100,fc) != NULL) {
	xtg_speak(s,4,"CLINE is:\n<%s>",cline);
	for (i=0;i<100;i++) {
	    if (cline[i] == 39) cline[i]=' ';
	}
	sscanf(cline,"%s %d %s",cname,reclen,ctype);
    }
    else{
	return (EOF);
    }



    u_eightletter(cname);
    xtg_speak(s,3,"cname is <%s>",cname);	
    xtg_speak(s,3,"reclen is %d",*reclen);	
    xtg_speak(s,3,"ctype is <%s>",ctype);	

    /* Now read the record itself */
    
    if (strcmp(ctype, "REAL") == 0) { 
	for (i=0;i<*reclen;i++) {
	    iok=fscanf(fc,"%f",&fvalue);
	    if (iok != 1) return iok;
	    tmp_float_v[i]=fvalue;
	}
	fgets(cline,100,fc);
    }
    else if (strcmp(ctype, "DOUB") == 0) { 
	for (i=0;i<*reclen;i++) {
	    /*
	     * Must read the value as a string, since C cannot(?)
	     * understand the Fortran ".30000000000000D+00" format
	     * Therefore I replace D with E
	     */
	    iok=fscanf(fc,"%s",sdvalue);
	    for (j=0;j<20;j++) {
		if (sdvalue[j] == 'D' || sdvalue[j] == 'd') sdvalue[j]='E';
	    }  
	    iok=sscanf(sdvalue,"%lf",&dvalue);
	    if (iok != 1) return iok;
	    tmp_double_v[i]=dvalue;
	}
	fgets(cline,100,fc);
    }

    else if (strcmp(ctype, "INTE") == 0) { 
	for (i=0;i<*reclen;i++) {
	    iok=fscanf(fc,"%d",&ivalue);
	    if (iok != 1) return iok;
	    tmp_int_v[i]=ivalue;
	}
	fgets(cline,100,fc);
    }

    /* CHAR need special tratment; must get rid of "'" */

    else if (strcmp(ctype, "CHAR") == 0) { 
	/* scan each line; never more than reclen lines */
	irec=0;
	for (jline=0;jline<*reclen;jline++) {
	    if (fgets(cline,100,fc) != NULL) {
		/* looking for 8 items, which I think is MAX. If less
		 * then iok will tell me so! Another problem is that
		 * fields may be EMPTY, but still they should be counted
		 * as an empty field (e.g. '       '). Therefore, I must
		 * first fill spaces with .... and then remove the "'"
		 */
		for (j=0; j<100; j++) {
		    /* the ASCII value of ' is 39 */ 
		    if (cline[j] == 39) {
			cline[j]=' '; /* removing ' at start of string */
			for (jj=j+1; jj<j+9; jj++) {
			    if (cline[jj] == ' ') cline[jj] = '.';
			}
			cline[j+9]=' '; /* removing ' at end of string */
		    }
		}
		iok=sscanf(cline,"%s%s%s%s%s%s%s%s",
			   cv[0],cv[1],cv[2],cv[3],cv[4],cv[5],cv[6],cv[7]);
		xtg_speak(s,4,"Numbers of words is: %d",iok);
		for (m=0;m<iok;m++) {
		    xtg_speak(s,4,"cv %d is: <%s>",m, cv[m]);
		    u_eightletter(cv[m]);
		    xtg_speak(s,4,"cv %d is: <%s>",m, cv[m]);
		    strcpy(tmp_string_v[irec], cv[m]);
		    irec++;
		    if (irec > *reclen) break;
		}
		if (irec > *reclen) break;
	    }
	}
	if (debug > 3) {
	    for (m=0;m<*reclen;m++) xtg_speak(s,4,"tmp_string %d is: <%s>",
					   m, tmp_string_v[m]);
	}


    }

    else if (strcmp(ctype, "LOGI") == 0) { 
	for (i=0;i<*reclen;i++) {
	    iok=fscanf(fc,"%s",&cvalue);
	    if (iok != 1) return iok;
	    if (cvalue == 'F') ivalue=0;
	    if (cvalue == 'T') ivalue=1;
	    tmp_logi_v[i]=ivalue;
	}
	fgets(cline,100,fc);
    }

    else {
	exit(23);
    }
	    
    return 0;
}

