/*
 * ############################################################################
 * read_eclipse_binary_record.c
 * Reads single Eclipse output records
 * Author: JCR
 * ############################################################################
 *
 *
 * ############################################################################
 */

#include <string.h>
#include <stdlib.h>
#include "libxtg.h"
#include "libxtg_.h"

/*
 * ****************************************************************************
 *                     READ_ECLIPSE_BINARY_RECORD
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
 * ----------------------------------------------------------------------------
 *
 */


int u_read_ecl_bin_record (
			   char    *cname,
			   char    *ctype,
			   int     *reclen,
			   int     max_alloc_int,
			   int     max_alloc_float,
			   int     max_alloc_double,
			   int     max_alloc_char,
			   int     max_alloc_logi,
			   int     *tmp_int_v,
			   float   *tmp_float_v,
			   double  *tmp_double_v,
			   char    **tmp_string_v,
			   int     *tmp_logi_v,
			   FILE    *fc,
			   int     debug
			   )

/*
 * This routine reads the Eclipse standard binary format
 */

{
    int    ftn_1, ftn_2, ftn_reclen, nrecord, swap, myint;
    int    i, ic, istat=1, reclenx;
    float  myfloat;
    double mydouble;
    char mystring[9];
    char   s[24]="u_read_ecl_bin_record";

    swap=0;
    // Intel based systems use little endian
    if (x_swap_check()==1) swap=1;


    /* read the description line, as e.g.:
       <16>'CORNERS '          24 'REAL'<16>
       where the <16> are 4 byte int determining record
       length (8+4+4 = 16 bytes)
    */

    istat=fread(&myint,4,1,fc);
    if (swap) SWAP_INT(myint); ftn_1=myint;
    if (istat != 1) {
	if (istat == EOF) {
	    return EOF;
	}
	else{
	    return -233222;
	}
    }

    xtg_speak(s,4,"ftn_1 is %d",ftn_1);
    if ((istat=fread(cname,8,1,fc)) != 1) return -233222;
    cname[8]='\0';
    xtg_speak(s,3,"cname is <%s>",cname);
    istat=fread(&myint,4,1,fc); if (swap) SWAP_INT(myint); *reclen=myint;
    if (istat != 1) return -233222;
    xtg_speak(s,4,"reclen is %d",*reclen);

    /* this is the total number of records in each entry */
    reclenx=*reclen;

    if ((istat=fread(ctype,4,1,fc)) != 1) return -233222;;
    ctype[4]='\0';
    xtg_speak(s,3,"ctype is <%s>",ctype);

    istat=fread(&myint,4,1,fc); if (swap) SWAP_INT(myint); ftn_2=myint;


    if (istat != 1) return -233222;
    xtg_speak(s,4,"ftn_2 is %d",ftn_2);


    /* check allocation */
    if (strcmp(ctype, "INTE") == 0 && reclenx > max_alloc_int) {
	xtg_error(s,"Allocation error (INTE) !. reclenx = %d vs max_alloc = %d  STOP!",reclenx,max_alloc_int);
    }

    if (strcmp(ctype, "REAL") == 0 && reclenx > max_alloc_float) {
	xtg_error(s,"Allocation error (REAL) !. reclenx = %d vs max_alloc = %d  STOP!",reclenx,max_alloc_float);
    }

    if (strcmp(ctype, "DOUB") == 0 && reclenx > max_alloc_double) {

	xtg_error(s,"Allocation error (DOUBLE) !. reclenx = %d vs max_alloc = %d  STOP!",reclenx,max_alloc_double);

	/* /\* in this, max_alloc_double must be updtaed to the caller *\/ */

	/* xtg_speak(s,2,"Need to realloc memory ... reclenx=%d vs max_alloc=%d  STOP!",reclenx,max_alloc_double); */
	/* tmp_double_v = realloc(tmp_double_v,reclenx*sizeof(double)) ; */
	/* if (tmp_double_v != NULL) { */
	/*     xtg_speak(s,2,"Realloc memory ... OK"); */
	/* } */
	/* else{ */
	/*     xtg_error(s,"Reallocation error (DOUBLE)! STOP!"); */
	/* } */

    }

    if (strcmp(ctype, "LOGI") == 0 && reclenx > max_alloc_logi) {
	xtg_error(s,"Allocation error (LOGI) !. reclenx = %d vs max_alloc = %d  STOP!",reclenx,max_alloc_logi);
    }


    if (strcmp(ctype, "CHAR") == 0 && reclenx > max_alloc_char) {
	xtg_error(s,"Allocation error (CHAR) !. reclenx = %d vs max_alloc = %d  STOP!",reclenx,max_alloc_char);;
    }




    /* Now read the record itself */

    ic=0;
    if (strcmp(ctype, "REAL") == 0) {
	xtg_speak(s,3,"ctype is REAL");
	while (*reclen > 0) {
	    /* ftn_1 will tell us how long the record is (in bytes) */
	    istat=fread(&myint,4,1,fc); if (swap) SWAP_INT(myint); ftn_1=myint;
	    if (istat != 1) return -233222;
	    ftn_reclen=(ftn_1/sizeof(float));
	    if (ftn_reclen < *reclen) { nrecord=ftn_reclen; } else {nrecord=*reclen;}



	    /* read the float array */
	    for (i=0;i<nrecord;i++) {
		istat=fread(&myfloat,4,1,fc); if (swap) SWAP_FLOAT(myfloat);
		tmp_float_v[ic++]=myfloat;
	    }
	    /* end of record integer: */
	    istat=fread(&myint,4,1,fc); if (swap) SWAP_INT(myint); ftn_2=myint;
	    /* remaining record length */
	    *reclen=*reclen-nrecord;
	    xtg_speak(s,4,"Remaining reclen is %d",*reclen);
	}
    }
    else if (strcmp(ctype, "DOUB") == 0) {
	xtg_speak(s,3,"ctype is DOUB");
	while (*reclen > 0) {
	    /* ftn_1 will tell us how long the record is (in bytes) */
	    istat=fread(&myint,4,1,fc); if (swap) SWAP_INT(myint); ftn_1=myint;
	    if (istat != 1) return -233222;
	    ftn_reclen=(ftn_1/sizeof(double));
	    if (ftn_reclen < *reclen) { nrecord=ftn_reclen; } else {nrecord=*reclen;}


	    /* read the double array */
	    for (i=0;i<nrecord;i++) {
		istat=fread(&mydouble,8,1,fc); if (swap) SWAP_DOUBLE(mydouble);
		tmp_double_v[ic++]=mydouble;
	    }
	    /* end of record integer: */
	    istat=fread(&myint,4,1,fc); if (swap) SWAP_INT(myint); ftn_2=myint;
	    /* remaining record length */
	    *reclen=*reclen-nrecord;
	    xtg_speak(s,4,"Remaining reclen is %d",*reclen);
	}
    }

    else if (strcmp(ctype, "INTE") == 0) {
	xtg_speak(s,3,"ctype is INTE");
	while (*reclen > 0) {
	    /* ftn_1 will tell us how long the record is (in bytes) */
	    istat=fread(&myint,4,1,fc); if (swap) SWAP_INT(myint); ftn_1=myint;
	    if (istat != 1) return -233222;
	    ftn_reclen=(ftn_1/sizeof(int));
	    if (ftn_reclen < *reclen) { nrecord=ftn_reclen; } else {nrecord=*reclen;}

	    /* read the int array */
	    xtg_speak(s,3,"nrecord is %d",nrecord);


	    for (i=0;i<nrecord;i++) {
		istat=fread(&myint,4,1,fc); if (swap) SWAP_INT(myint);
		tmp_int_v[ic++]=myint;
	    }
	    /* end of record integer: */
	    istat=fread(&myint,4,1,fc); if (swap) SWAP_INT(myint); ftn_2=myint;
	    /* remaining record length */
	    *reclen=*reclen-nrecord;
	    xtg_speak(s,4,"Remaining reclen is %d",*reclen);
	}
    }

    /* no Byte swap needed for char */
    else if (strcmp(ctype, "CHAR") == 0) {
	xtg_speak(s,3,"ctype is CHAR");
	while (*reclen > 0) {
	    /* ftn_1 will tell us how long the record is (in bytes) */
	    istat=fread(&myint,4,1,fc); if (swap) SWAP_INT(myint); ftn_1=myint;
	    ftn_reclen=(ftn_1/(8*sizeof(char)));
	    if (ftn_reclen < *reclen) {nrecord=ftn_reclen;}else{nrecord=*reclen;}
	    if ((istat=fread(mystring,8,nrecord,fc))
		!= nrecord) return -233222;
	    /* end of record integer: */
	    istat=fread(&myint,4,1,fc); if (swap) SWAP_INT(myint); ftn_2=myint;

	    /* remaining record length */
	    *reclen = *reclen-nrecord;
	    xtg_speak(s,4,"Remaining reclen is %d",*reclen);
	}
    }

    else if (strcmp(ctype, "LOGI") == 0) {
	xtg_speak(s,3,"ctype is LOGI");
	while (*reclen > 0) {
	    /* ftn_1 will tell us how long the record is (in bytes) */
	    istat=fread(&myint,4,1,fc); if (swap) SWAP_INT(myint); ftn_1=myint;
	    if (istat != 1) return -233222;
	    ftn_reclen=(ftn_1/sizeof(int));
	    if (ftn_reclen < *reclen) { nrecord=ftn_reclen; } else {nrecord=*reclen;}


	    /* read the array */
	    for (i=0;i<nrecord;i++) {
		istat=fread(&myint,4,1,fc); if (swap) SWAP_INT(myint);
		tmp_logi_v[ic++]=myint;
	    }
	    /* end of record integer: */
	    istat=fread(&myint,4,1,fc); if (swap) SWAP_INT(myint); ftn_2=myint;
	    /* remaining record length */
	    *reclen=*reclen-nrecord;
	    xtg_speak(s,4,"Remaining reclen is %d",*reclen);
	}
    }

    /* not sure what this is...*/
    else if (strcmp(ctype, "MESS") == 0) {
	xtg_speak(s,3,"ctype is MESS");
	while (*reclen > 0) {
	    /* ftn_1 will tell us how long the record is (in bytes) */
	    istat=fread(&myint,4,1,fc); if (swap) SWAP_INT(myint); ftn_1=myint;
	    if (istat != 1) return -233222;
	    ftn_reclen=(ftn_1/sizeof(int));
	    if (ftn_reclen < *reclen) { nrecord=ftn_reclen; } else {nrecord=*reclen;}

	    /* read the array */
	    for (i=0;i<nrecord;i++) {
		istat=fread(&myint,4,1,fc); if (swap) SWAP_INT(myint);
		tmp_logi_v[ic++]=myint;
	    }
	    /* end of record integer: */
	    istat=fread(&myint,4,1,fc); if (swap) SWAP_INT(myint); ftn_2=myint;
	    /* remaining record length */
	    *reclen=*reclen-nrecord;
	    xtg_speak(s,4,"Remaining reclen is %d",*reclen);
	}
    }

    else {
	exit(23);
    }

    if (ic>0 && (ic != reclenx)) {
	xtg_error(s,"Something is rotten: IC is %d, RECLENX is %d",ic,reclenx);
    }

    *reclen = reclenx;
    return 0;
}
