/*
 * ############################################################################
 * grd3d_export_roff_prop
 * Exporting a Roff ASCII (if mode>0) of BINARY property to file
 * This routine must follow the grd3d_export_roff_pstart or
 * grd3d_export_roff_grid. The grd3d_export_roff_end must come after.
 * Author: JCR
 * ############################################################################
 * $Id: grd3d_export_roff_prop.c,v 1.1 2001/03/14 08:02:29 bg54276 Exp bg54276 $
 * $Source: /d/proj/bg/avresmod/src/gplib/GPLExt/RCS/grd3d_export_roff_prop.c,v $
 *
 * $Log: grd3d_export_roff_prop.c,v $
 * Revision 1.1  2001/03/14 08:02:29  bg54276
 * Initial revision
 *
 * Revision 1.1  2000/12/12 17:24:54  bg54276
 * Initial revision
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
 *                         ROFF FORMAT
 ****************************************************************************
 * The ROFF (Roxar Open File Format) is like this:
 *
 *roff-asc
 *#ROFF file#
 *#Creator: RMS - Reservoir Modelling System, version 6.0.2#
 *tag filedata
 *int byteswaptest 1
 *char filetype  "parameter"
 *char creationDate  "08/11/2000 10:45:57"
 *endtag
 *tag version
 *int major 2
 *int minor 0
 *endtag
 *tag dimensions
 *int nX 80
 *int nY 70
 *int nZ 20
 *endtag
 *tag parameter
 *char name  "aBase"
 *array float data 112000
 *     ...numbers...
 *endtag
 *tag eof
 *endtag
 *
 * ----------------------------------------------------------------------------
 *
 */


void grd3d_export_roff_prop_old (
                                 int     mode,
                                 int     nx,
                                 int     ny,
                                 int     nz,
                                 int     num_subgrds,
                                 int     isubgrd_to_export,
                                 int     *p_subgrd_v,
                                 char    *pname,
                                 char    *ptype,
                                 int     *p_int_v,
                                 double  *p_double_v,
                                 int     ncodes,
                                 char    **codenames,
                                 int     *codevalues,
                                 char    *filename,
                                 int     debug
                                 )


{

    int    nn, nz_true, nz1, nz2, i, j, k, kc;
    int    ib, i_tmp, ier=0;
    int    myint;
    float  myfloat;
    char   mybyte, mychar;
    char   mystring[ROFFSTRLEN];
    FILE   *fc;
    char   s[24]="gr.._ex.._roff_prop_old";

    xtgverbose(debug);

    xtg_speak(s,2,"==== Entering <export_grd3d_roff_prop> ====");


    xtg_speak(s,1,"PNAME is <%s>",pname);

    if (strcmp(ptype,"double")==0) ptype="float";

    /*
     *-------------------------------------------------------------------------
     * It is possible to export just one subgrid (if isubgrd_to_export >0)
     * Must do some calculations for this here:
     *-------------------------------------------------------------------------
     */
    nz_true=nz;
    nz1=1;
    nz2=nz;

    if (isubgrd_to_export <= num_subgrds) {
	if (isubgrd_to_export > 0) {
	    /* redefine nz */
	    nz_true=p_subgrd_v[isubgrd_to_export-1]; /*vector starts at 0 */

	    /* find nz1 and nz2 (counted from top) */
	    k=0;
	    for (kc=0;kc<(isubgrd_to_export-1);kc++) {
		k=k+p_subgrd_v[kc];
	    }
	    nz1=k+1;
	    nz2=k+p_subgrd_v[isubgrd_to_export-1];
	    xtg_speak(s,2,"Exporting subgrid, using K range: %d - %d",nz1,nz2);
	}
    }
    else{
	xtg_error(s,"Fatal error: isubgrd_to_export too large");
    }


    xtg_speak(s,2,"Opening ROFF file (append)...");
    if (mode==0) {
	fc=fopen(filename,"ab");
    }
    else{
	fc=fopen(filename,"ab");
    }

    xtg_speak(s,2,"Opening ROFF file ... DONE!");

    /*
     *-------------------------------------------------------------------------
     * Header of file exists by routines export_grd3d_roff_grid (ROFF
     * grid+prop files) or export_grd3d_roff_pstart (ROFF prop only files)
     *-------------------------------------------------------------------------
     */


    nn=nx*ny*nz_true;
    xtg_speak(s,2,"Exporting property: %s (%s)",pname,ptype);

    if (mode==1) {
	fprintf(fc,"tag parameter\n");
	fprintf(fc,"char name \"%s\"\n",pname);
    }
    else if (mode==0) {
	ier=fwrite("tag\0parameter\0",1,14,fc);
	ier=fwrite("char\0name\0",1,10,fc);
	for (i=0;i<=100;i++) {
	    mychar=pname[i];
	    ier=fwrite(&mychar,1,1,fc);
	    if (pname[i]=='\0') break;
	}
    }

    /*
     * A codeName/codeValues table may exist and must be printed
     */
    if (ncodes > 0) {
	if (mode>0 && mode <2) {
	    fprintf(fc,"array char codeNames %d\n",ncodes);
	    for (i=0;i<ncodes;i++) {
		for (j=0;j<33;j++) {
		    mystring[j]=codenames[i][j];
		    if (mystring[j]=='\0') break;
		}
		fprintf(fc,"\"%s\"\n",mystring);
	    }
	    i_tmp=1;
	    fprintf(fc,"array int codeValues %d\n",ncodes);
	    for (i=0;i<ncodes;i++) {
		fprintf(fc," %d",codevalues[i]);
		i_tmp++;
		if (i_tmp > 12) {
		    i_tmp=1;
		    fprintf(fc,"\n");
		}
	    }
	    if (i_tmp!=1)fprintf(fc,"\n");
	}
	else{
	    ier=fwrite("array\0char\0codeNames\0",21,1,fc);
	    myint=ncodes;
	    ier=fwrite(&myint,4,1,fc);
	    for (i=0;i<ncodes;i++) {
		for (j=0;j<33;j++) {
		    mystring[j]=codenames[i][j];
		    if (mystring[j]=='\0') break;
		}
		ier=fwrite(mystring,1,j+1,fc);
	    }
	    ier=fwrite("array\0int\0codeValues\0",21,1,fc);
	    myint=ncodes;
	    ier=fwrite(&myint,4,1,fc);
	    for (i=0;i<ncodes;i++) {
		myint=codevalues[i];
		ier=fwrite(&myint,4,1,fc);
	    }
	}
    }

    /*
     * The array itself...

     */
    if (mode==1) {
	fprintf(fc,"array %s data %d\n",ptype,nn);
    }
    else if (mode==0) {
	/* doubble exported as floats */
	if (strcmp(ptype,"double")==0 || strcmp(ptype,"float")==0) {
	    ier=fwrite("array\0float\0data\0",1,17,fc);
	}
	else if (strcmp(ptype,"int")==0) {
	    ier=fwrite("array\0int\0data\0",1,15,fc);
	}
	else if (strcmp(ptype,"byte")==0) {
	    ier=fwrite("array\0byte\0data\0",1,16,fc);
	}
	myint=nn;
	ier=fwrite(&myint,4,1,fc);
    }


    i_tmp=1;
    for (i=1;i<=nx;i++) {
	for (j=1;j<=ny;j++) {
	    for (k=nz2;k>=nz1;k--) {
		ib=x_ijk2ib(i,j,k,nx,ny,nz,0);
		/* double is exported as float */
		if (strcmp(ptype,"float")==0) {
		    if (mode>0) {
			fprintf(fc,"  %e",p_double_v[ib]);
			i_tmp++;
			if (i_tmp > 6) {
			    i_tmp=1;
			    fprintf(fc,"\n");
			}
		    }
		    else{
			myfloat=p_double_v[ib];
			ier=fwrite(&myfloat,4,1,fc);
		    }
		}
		else if (strcmp(ptype,"int")==0) {
		    if (mode>0) {
			fprintf(fc,"  %d",p_int_v[ib]);
			i_tmp++;
			if (i_tmp > 6) {
			    i_tmp=1;
			    fprintf(fc,"\n");
			}
		    }
		    else{
			myint=p_int_v[ib];
			ier=fwrite(&myint,4,1,fc);
		    }
		}
		else if (strcmp(ptype,"byte")==0) {
		    if (p_int_v[ib] == UNDEF_ROFFINT) {
			mybyte=UNDEF_ROFFBYTE;
		    }
		    else{
			mybyte=p_int_v[ib];
		    }
		    if (mode>0) {
			fprintf(fc,"  %d",mybyte);
			i_tmp++;
			if (i_tmp > 6) {
			    i_tmp=1;
			    fprintf(fc,"\n");
			}
		    }
		    else{
			ier=fwrite(&mybyte,1,1,fc);
		    }
		}
	    }
	}
    }

    if (mode==1) {
	if (i_tmp!=1)fprintf(fc,"\n");
	fprintf(fc,"endtag\n");
    }
    else if (mode==0) {
	ier=fwrite("endtag\0",1,7,fc);
    }

    xtg_speak(s,2,"Writing property...DONE!");

    fclose(fc);

    xtg_speak(s,2,"Exit from export roff");
}
