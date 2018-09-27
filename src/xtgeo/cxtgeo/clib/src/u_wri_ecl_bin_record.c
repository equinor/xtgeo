/*
 * ############################################################################
 * u_wri_ecl_bin_record.c
 * Utility sub routine:
 * Writes single Eclipse output records on binary format
 * TODO: Not sure on how to handle string records
 * Author: JCR
 * ############################################################################
 * $Id: $ 
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
 *                     WRITE_ECLIPSE_BINARY_RECORD
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
 * For the binary form, the record starts and ends with a 4 byte integer, that
 * says how long the current record is, in bytes.
 *
 * writes the description line, as e.g.:
 *      <16>'CORNERS '          24 'REAL'<16>
 *      where the <16> are 4 byte int determining record
 *     length (8+4+4 = 16 bytes)
 *
 * ----------------------------------------------------------------------------
 *
 */   


int u_wri_ecl_bin_record (
			  char    *cname,
			  char    *ctype,
			  int     reclen,
			  int     *tmp_int_v,
			  float   *tmp_float_v,
			  double  *tmp_double_v,
			  char    **tmp_string_v,
			  int     *tmp_logi_v,
			  FILE    *fc,
			  int     debug
			  )
     
/*
 * This routine writes the Eclipse standard binary format
 */

{
    int    i, j, myint, mylogi, nrec, *nblock, numrec=0;
    int    jj, swap, ier;
    float  myfloat;
    double mydouble;
    char   s[24]="u_wri_ecl_bin_record";

 
    /* max 100,000 blocks, which is 100 mill cells... */ 
    nblock=calloc(100000,sizeof(int));

    if (debug >=4 ) {
	xtg_speak(s,4,"Entering <write_eclipse_binary_record>...");
	xtg_speak(s,4,"Writing <%s> of type <%s>",cname, ctype);
    }

    swap=0;
    // Intel based systems use little endian
    if (x_swap_check()==1) swap=1;

    /*
     *=========================================================================
     * Record header is quite standard
     *=========================================================================
     */

    myint=16;    
    if (swap) SWAP_INT(myint); ier=fwrite(&myint,4,1,fc);
    ier=fwrite(cname,1,8,fc);
    myint=reclen;
    if (swap) SWAP_INT(myint); ier=fwrite(&myint,4,1,fc);
    ier=fwrite(ctype,1,4,fc);
    myint=16;    
    if (swap) SWAP_INT(myint); ier=fwrite(&myint,4,1,fc);



    /* 
     * Data: Fortran has limited size of block within a record 
     */
    nrec=reclen;
    for (i=0;i<100000;i++) {
	if (nrec > ECLNUMBLOCKLEN) {
	    nblock[i]=ECLNUMBLOCKLEN;
	    nrec=nrec-ECLNUMBLOCKLEN;
	}
	else{
	    nblock[i]=nrec;
	    numrec=i+1;
	    break;
	}
    }

    xtg_speak(s,4,"Number of blocks within record: %d", numrec);

    /*
     *=========================================================================
     * Data
     *=========================================================================
     */



    jj=0;
    if (strcmp(ctype, "REAL") == 0) { 
	for (i=0;i<numrec;i++) {
	    myint=nblock[i]*4;
	    if (swap) SWAP_INT(myint); ier=fwrite(&myint,4,1,fc);
	    for (j=0;j<nblock[i];j++) {
		myfloat=tmp_float_v[jj++];
		if (swap) SWAP_FLOAT(myfloat); ier=fwrite(&myfloat,4,1,fc);
	    }
	    ier=fwrite(&myint,4,1,fc); /* its already swapped ... */
	}

    }
    
    else if (strcmp(ctype, "DOUB") == 0) { 
	for (i=0;i<numrec;i++) {
	    myint=nblock[i]*8;
	    if (swap) SWAP_INT(myint); ier=fwrite(&myint,4,1,fc);
	    for (j=0;j<nblock[i];j++) {
		mydouble=tmp_double_v[jj++];
		if (swap) SWAP_DOUBLE(mydouble); ier=fwrite(&mydouble,8,1,fc);
	    }
	    ier=fwrite(&myint,4,1,fc); /* its already swapped ... */
	}

    }

    else if (strcmp(ctype, "INTE") == 0) { 
	for (i=0;i<numrec;i++) {
	    myint=nblock[i]*4;
	    if (swap) SWAP_INT(myint); ier=fwrite(&myint,4,1,fc);
	    for (j=0;j<nblock[i];j++) {
		myint=tmp_int_v[jj++];
		if (swap) SWAP_INT(myint); ier=fwrite(&myint,4,1,fc);
	    }
	    myint=nblock[i]*4;
	    if (swap) SWAP_INT(myint); ier=fwrite(&myint,4,1,fc);
	}
    }

    /* ??
    else if (strcmp(ctype, "CHAR") == 0) { 
	for (i=0;i<numrec;i++) {
	    myint=nblock[i]*9;
	    ier=fwrite(&myint,4,1,fc);
	    for (j=0;j<nblock[i];j++) {
		ier=fwrite(tmp_int_v[jj++],4,1,fc);
	    }
	    ier=fwrite(&myint,4,1,fc);
	}
	myint=reclen*8;
	ier=fwrite(&myint,4,1,fc);
	for (i=0; i<reclen; i++) {
	    ier=fwrite(tmp_string_v[i],8,1,fc);
	}
	ier=fwrite(&myint,4,1,fc);
    }
    ?? */

    else if (strcmp(ctype, "LOGI") == 0) { 
	for (i=0;i<numrec;i++) {
	    myint=nblock[i]*4;
	    if (swap) SWAP_INT(myint); ier=fwrite(&myint,4,1,fc);
	    for (j=0;j<nblock[i];j++) {
		mylogi=tmp_logi_v[jj++];
		if (swap) SWAP_INT(mylogi); ier=fwrite(&mylogi,4,1,fc);
	    }
	    ier=fwrite(&myint,4,1,fc);
	}
    }

    else {
	exit(23);
    }
	    
    free(nblock);

    if (debug >=4 ) {
	xtg_speak(s,4,"Exit from <write_eclipse_binary_record>...");
    }
    return 0;
}





