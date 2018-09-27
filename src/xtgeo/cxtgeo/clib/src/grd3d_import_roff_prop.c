/*
 * ############################################################################
 * grd3d_import_roff_prop.c
 * Reading a Roff ASCII property
 * Author: JCR
 * ############################################################################
 * $Id: grd3d_import_roff_prop.c,v 1.1 2001/03/14 08:02:29 bg54276 Exp bg54276 $
 * $Source: /d/proj/bg/avresmod/src/gplib/GPLExt/RCS/grd3d_import_roff_prop.c,v $
 *
 * $Log: grd3d_import_roff_prop.c,v $
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
 *                      IMPORT_GRD3D_ROFF_PROP
 ****************************************************************************
 * This routine is based on that every thing regarding the GRID is known
 * (geometry, nx, ny, nz, ...)
 * ----------------------------------------------------------------------------
 * It looks for one property in the file and imports
 *
 *
 */


void grd3d_import_roff_prop (
			      int     nx,
			      int     ny,
			      int     nz,
			      char    *prop_name,
			      int     *p_int_v,
			      double  *p_dfloat_v,
			      char    **codenames,
			      int     *codevalues,
			      char    *filename,
			      int     debug
			      )


{
    char   cname[ROFFSTRLEN];
    int    iok;
    FILE   *fc;
    char   sub[24]="grd3d_import_roff_prop";

    xtgverbose(debug);
    xtg_speak(sub,2,"==== Entering grd3d_import_roff_prop ====");
    xtg_speak(sub,2,"Looking for <%s>",prop_name);




    /*
     *-------------------------------------------------------------------------
     * Open file
     *-------------------------------------------------------------------------
     */

    xtg_speak(sub,2,"Opening ROFF file...");
    fc=fopen(filename,"r");
    if (fc==NULL) {
	xtg_error(sub,"Cannot open file!");
    }
    xtg_speak(sub,2,"Opening ROFF file...OK!");

    /*
     *-------------------------------------------------------------------------
     * Read first few bytes to determine ASCII or BINARY
     *-------------------------------------------------------------------------
     */

    iok=fread(cname,7,1,fc);
    if (iok != 1) xtg_error(sub, "Error in fread");
    fclose(fc);
    cname[7]='\0';
    xtg_speak(sub,2,"Header is %s\n", cname);

    if (strcmp(cname,"roff-as")==0) {
	xtg_speak(sub,2,"ROFF ASCII file...");
	_grd3d_imp_roff_asc_prp(
				nx,
				ny,
				nz,
				prop_name,
				p_int_v,
				p_dfloat_v,
				codenames,
				codevalues,
				filename,
				debug
				);

    }
    else{
	xtg_speak(sub,2,"ROFF BINARY file...");
	_grd3d_imp_roff_bin_prp(
				nx,
				ny,
				nz,
				prop_name,
				p_int_v,
				p_dfloat_v,
				codenames,
				codevalues,
				filename,
				debug
				);
    }
}


void _grd3d_imp_roff_asc_prp(
			     int     nx,
			     int     ny,
			     int     nz,
			     char    *prop_name,
			     int     *p_int_v,
			     double  *p_dfloat_v,
			     char    **codenames,
			     int     *codevalues,
			     char    *filename,
			     int     debug
			     )

{
    int    i, j, k, ipos, iok, line, num;
    char   carray[ROFFSTRLEN], ctype[ROFFSTRLEN];
    char   cname[ROFFSTRLEN], cdum[ROFFSTRLEN], cline[133];
    char   mystring[ROFFSTRLEN];
    int    probablyfound, propstatus;

    int    ivalue, mylen, myint;
    float  fvalue;
    char   sub[24]="_grd3d_imp_roff_asc_prp";

    FILE   *fc;

    xtgverbose(debug);

    propstatus=-1;

    i = 0;
    /*
     *-------------------------------------------------------------------------
     * Open file
     *-------------------------------------------------------------------------
     */

    xtg_speak(sub,2,"Opening ROFF file...");
    fc=fopen(filename,"r");
    if (fc==NULL) {
	xtg_error(sub,"Cannot open file!");
    }
    xtg_speak(sub,2,"Opening ROFF file...OK!");



    /*
     *=========================================================================
     * Loop file... It is NOT necessary to do many tests; that should be done
     * by the calling PERL script?
     *=========================================================================
     */


    for (line=1;line<99999999;line++) {

	/* Get offsets */
	x_fgets(cline,132,fc);
	xtg_speak(sub,4,"CLINE is:\n%s",cline);

	/*
	 *---------------------------------------------------------------------
	 * Getting 'parameter' values
	 *---------------------------------------------------------------------
	 */
	if (strncmp(cline, "tag parameter", 13) == 0) {
	    xtg_speak(sub,3,"Tag parameter was found");
	    x_fgets(cline,132,fc);
	    iok=sscanf(cline,"%s %s %s",ctype, cdum, cname);
	    xtg_speak(sub,2,"Property is %s",cname);
	    xtg_speak(sub,3,"IOK is: ",iok);

	    /*
	     * Due to a BUG in RMS IPL RoffExport, the parameter (prop) name
	     * may be "" (empty).
	     * In such cases, it is assumed to be correct!
	     * Must rely on structured programming
	     */
	    probablyfound=0;
	    if (strcmp(cname,"\"\"")==0) {
		probablyfound=1;
		xtg_speak(sub,2,"Assume that empty parameter name means: %s",prop_name);
	    }
	    /* make fnutts around string: */
	    sprintf(mystring,"\"%s\"",prop_name);
	    if (probablyfound==1 || strcmp(cname,mystring) == 0) {
		x_fgets(cline,132,fc);
		xtg_speak(sub,2,"Now reading property: %s ...", prop_name);

		propstatus=0;
		sscanf(cline,"%s",ctype);
		if (strcmp(ctype,"array")==0) {
		    sscanf(cline,"%s %s %s %d",carray, ctype, cname, &num);
		    xtg_speak(sub,3,"CLINE0 is:\n%s",cline);
		    /*
		     * For discrete variables, codeNames and codeValues is present,
		     * either as array of scalar!
		     */
		    if (strcmp(cname,"codeNames")==0) {
			for (i=0; i<num;  i++) {
			    if (i==(num-1)){
				iok=fscanf(fc,"%s\n",mystring);
			    }
			    else{
				iok=fscanf(fc,"%s",mystring);
			    }
			    /* remove fnutts: */
			    mylen=strlen(mystring);
			    if (mylen>0) {
				for (j=1;j<mylen;j++) {
				    codenames[i][j-1]=mystring[j];
				}
				codenames[i][mylen-2]='\0';
			    }
			    xtg_speak(sub,3,"codeName %d is <%s>",i,codenames[i]);
			}
			/* When codeNames is present, codeValues is next! */
			x_fgets(cline,132,fc);
			xtg_speak(sub,4,"CLINE is:\n%s",cline);
			sscanf(cline,"%s %s %s %d",carray, ctype, cname, &num);
			for (i=0; i<num;  i++) {
			    iok=fscanf(fc,"%d",&myint);
			    codevalues[i]=myint;
			    xtg_speak(sub,3,"codeValues %d is <%d>",i,codevalues[i]);
			}
			iok=fscanf(fc,"\n");

			x_fgets(cline,132,fc);
			xtg_speak(sub,3,"CLINE2 is:\n%s",cline);
			sscanf(cline,"%s %s %s %d",carray, ctype, cname, &num);

		    }


		}
		else{
		    /* codenames and codevalues given in scalar form */
		    sscanf(cline,"%s %s %s",ctype, cname, mystring);
		    /* remove fnutts: */
		    mylen=strlen(mystring);
		    if (mylen>0) {
			for (j=1;j<mylen;j++) {
			    codenames[0][j-1]=mystring[j];
			}
			codenames[0][mylen-2]='\0';
		    }
		    xtg_speak(sub,3,"codeName %d is <%s>",i,codenames[i]);

		    /* When codeNames is present, codeValues is next! */
		    x_fgets(cline,132,fc);
		    xtg_speak(sub,3,"CLINE1 is:\n%s",cline);
		    sscanf(cline,"%s %s %d",ctype, cname, &myint);
		    codevalues[0]=myint;
		    xtg_speak(sub,3,"codeValues %d is <%d>",i,codevalues[0]);
		    iok=fscanf(fc,"\n");

		    x_fgets(cline,132,fc);
		    xtg_speak(sub,3,"CLINE2 is:\n%s",cline);
		    sscanf(cline,"%s %s %s %d",carray, ctype, cname, &num);
		}



		xtg_speak(sub,3,"Reading array of type: <%s>",ctype);
		for (i=0;i<nx;i++) {
		    for (j=0;j<ny;j++) {
			for (k=0;k<nz;k++) {
			    if (strcmp(ctype,"int")==0 ||
				strcmp(ctype,"byte")==0 ) {

				iok=fscanf(fc,"%d",&ivalue);
				/* map directly to XTGeo form */
				ipos=(nz-(k+1))*ny*nx + j*nx + i;
				p_int_v[ipos]=ivalue;
				if (strcmp(ctype,"byte")==0 &&
				    p_int_v[ipos]==UNDEF_ROFFBYTE) {

				    p_int_v[ipos]=UNDEF_INT;

				}
			    }
			    else if (strcmp(ctype,"float")==0) {
				iok=fscanf(fc,"%f",&fvalue);
				/* map directly to XTGeo form */
				ipos=(nz-(k+1))*ny*nx + j*nx + i;
				p_dfloat_v[ipos]=fvalue;
				if (p_dfloat_v[ipos]==UNDEF_ROFFFLOAT) {
				    p_dfloat_v[ipos]=UNDEF;
				}

			    }
			}
		    }
		}
		xtg_speak(sub,2,"Reading property: %s ... DONE!", prop_name);
		break;
	    }
	}
    }

    fclose(fc);


    if (propstatus<0) {
	xtg_error(sub,"Requested property <%s> not found!",prop_name);
    }


    xtg_speak(sub,2,"==== Exiting grd3d_import_roff_prop ====");
}



void _grd3d_imp_roff_bin_prp (
			      int     nx,
			      int     ny,
			      int     nz,
			      char    *prop_name,
			      int     *p_int_v,
			      double  *p_dfloat_v,
			      char    **codenames,
			      int     *codevalues,
			      char    *filename,
			      int     debug
			      )
{
    int   aint, storevalue;
    char  abyte;
    float afloat;
    int   probablyfound;

    int   i, j, k, n, ipos, idum, ncodes, iok;
    int   swap;
    char  cname[ROFFSTRLEN], ctype[ROFFSTRLEN];
    char   sub[24]="_grd3d_imp_roff_bin_prp";
    int   propstatus=-1;

    FILE   *fc;


    /*
     *-------------------------------------------------------------------------
     * Open file
     *-------------------------------------------------------------------------
     */

    xtg_speak(sub,2,"Opening ROFF file...");
    fc=fopen(filename,"rb");
    if (fc==NULL) {
	xtg_error(sub,"Cannot open file!");
    }
    xtg_speak(sub,2,"Opening ROFF file...OK!");

    swap=0;
    if (x_byteorder(-1)>1) swap=1;

    /*
     *=========================================================================
     * Loop file...
     *=========================================================================
     */


    _grd3d_roffbinstring(cname, fc);
    for (idum=1;idum<999999;idum++) {

	_grd3d_roffbinstring(cname, fc);
	xtg_speak(sub,4,"Reading: %s",cname);


	if (strcmp(cname, "tag") == 0) {
	    _grd3d_roffbinstring(cname, fc);
	    /*
	     *-----------------------------------------------------------------
	     * Getting 'parameter' values
	     * Needs reprogramming ... not very elegant ...
	     *-----------------------------------------------------------------
	     */
	    if (strcmp(cname, "parameter") == 0) {
		xtg_speak(sub,2,"Tag parameter was found");
		_grd3d_roffbinstring(cname,fc); /*char*/
		_grd3d_roffbinstring(cname,fc); /*name*/
		_grd3d_roffbinstring(cname,fc); /*actual property name*/

		/*
		 * Due to a BUG in RMS IPL RoffExport, the property name
		 * may be "" (empty).
		 * In such cases, it is assumed to be correct!
		 * Must rely on structured programming
		 */
		probablyfound=0;
		if (strcmp(cname,"")==0) probablyfound=1;
		if (strcmp(prop_name,"generic")==0) probablyfound=1;

		if (probablyfound || strcmp(cname,prop_name)==0) {
		    xtg_speak(sub,3,"<%s> found!",cname);
		    storevalue=1;
		    propstatus=0;
		}
		else{
		    storevalue=0;
		}
		_grd3d_roffbinstring(cname,fc); /*array or...*/
		if (strcmp(cname,"array")==0) {
		    _grd3d_roffbinstring(ctype,fc); /*int or float or ...*/
		    _grd3d_roffbinstring(cname,fc); /*data or codeNames or ...?*/
		    if (strcmp(cname,"data")==0 && strcmp(ctype,"float")==0) {
			iok=fread(&n,4,1,fc);
			if (iok != 1) xtg_error(sub, "Error in reading");

			if (swap==1) SWAP_INT(n);
			for (i=0;i<nx;i++) {
			    for (j=0;j<ny;j++) {
				for (k=0;k<nz;k++) {
				    iok=fread(&afloat,4,1,fc);
				    if (swap==1) SWAP_FLOAT(afloat);
				    /* map directly to XTGeo form */
				    if (storevalue==1) {
					ipos=(nz-(k+1))*ny*nx + j*nx + i;
					p_dfloat_v[ipos]=afloat;
					if (p_dfloat_v[ipos]==UNDEF_ROFFFLOAT) {
					    p_dfloat_v[ipos]=UNDEF;
					}

				    }
				}
			    }
			}
			fclose(fc);
			return;
		    }
		    else if (strcmp(cname,"codeNames")==0) {
			xtg_speak(sub,3,"codeNames found: ",cname);
			/* get the number of codes */
			iok=fread(&ncodes,4,1,fc);
			if (swap==1) SWAP_INT(ncodes);
			xtg_speak(sub,3,"Number of codes: %d",ncodes);
			/*get the codenames*/
			if (storevalue==1) {
			    _grd3d_getchararray(codenames,ncodes,fc);
			}
			else{
			    /* read it dummy */
			    for (i=0; i<ncodes; i++) {
				_grd3d_roffbinstring(cname,fc);
			    }
			}
			/* getting array */
			_grd3d_roffbinstring(cname,fc); /*array ...*/
			_grd3d_roffbinstring(ctype,fc); /*int ...*/
			_grd3d_roffbinstring(cname,fc); /*codeValues*/
			iok=fread(&ncodes,4,1,fc);
			if (swap==1) SWAP_INT(ncodes);
			if (strcmp(ctype,"int")==0) {
			    _grd3d_getintarray(codevalues,ncodes,fc);
			}


			/* now get to the point */
			_grd3d_roffbinstring(cname,fc); /*array ...*/
			_grd3d_roffbinstring(ctype,fc); /*int .. byte ...*/
			_grd3d_roffbinstring(cname,fc); /*data*/

			/* currently, I am storing a byte as integer */

			if (strcmp(cname,"data")==0 && strcmp(ctype,"byte")==0) {
			    iok=fread(&n,4,1,fc);
			    if (swap==1) SWAP_INT(n);
			    for (i=0;i<nx;i++) {
				for (j=0;j<ny;j++) {
				    for (k=0;k<nz;k++) {
					iok=fread(&abyte,1,1,fc);
					/* map directly to XTGeo form */
					if (storevalue==1) {
                                            ipos=(nz-(k+1))*ny*nx + j*nx + i;
                                            p_int_v[ipos]=abyte;
                                            if (p_int_v[ipos]==UNDEF_ROFFBYTE) {
                                                p_int_v[ipos]=UNDEF_ROFFINT;
                                            }
					}
				    }
				}
			    }
			    fclose(fc);
			    return;
			}
			if (strcmp(cname,"data")==0 && strcmp(ctype,"int")==0) {
			    iok=fread(&n,4,1,fc);
			    if (swap==1) SWAP_INT(n);
			    for (i=0;i<nx;i++) {
				for (j=0;j<ny;j++) {
				    for (k=0;k<nz;k++) {
					iok=fread(&aint,4,1,fc);
					if (swap==1) SWAP_INT(aint);
					if (storevalue==1) {
					    /* map directly to XTGeo form */
					    ipos=(nz-(k+1))*ny*nx + j*nx + i;
					    p_int_v[ipos]=aint;
					}
				    }
				}
			    }
			    fclose(fc);
			    return;
			}
		    }
		}
		else{
		    _grd3d_roffbinstring(cname,fc); /*codeNames ...*/
		    xtg_speak(sub,3,"codeNames found ...: ",cname);
		    ncodes=1;
		    xtg_speak(sub,3,"Number of codes: %d",ncodes);
		    /*get the codenames*/
		    if (storevalue==1) {
			_grd3d_roffbinstring(cname,fc); /*value ...*/
			strcpy(codenames[0],cname);
			_grd3d_getchararray(codenames,ncodes,fc);
		    }
		    else{
			/* read dummy */
			_grd3d_roffbinstring(cname,fc); /*value ...*/
		    }
		    _grd3d_roffbinstring(cname,fc); /*int ...*/
		    _grd3d_roffbinstring(cname,fc); /*codeValues ...*/
		    iok=fread(&ncodes,4,1,fc);
		    if (swap==1) SWAP_INT(ncodes);
		    codevalues[0]=ncodes;

		    /* now get to the point */
		    _grd3d_roffbinstring(cname,fc); /*array ...*/
		    _grd3d_roffbinstring(ctype,fc); /*int .. byte ...*/
		    _grd3d_roffbinstring(cname,fc); /*data*/

		    /* currently, I am storing a byte as integer */

		    if (strcmp(cname,"data")==0 && strcmp(ctype,"byte")==0) {
			iok=fread(&n,4,1,fc);
			if (swap==1) SWAP_INT(n);
			for (i=0;i<nx;i++) {
			    for (j=0;j<ny;j++) {
				for (k=0;k<nz;k++) {
				    iok=fread(&abyte,1,1,fc);
				    /* map directly to XTGeo form */
				    if (storevalue==1) {
					ipos=(nz-(k+1))*ny*nx + j*nx + i;
					p_int_v[ipos]=abyte;
					if (p_int_v[ipos]==UNDEF_ROFFBYTE) {
					    //p_int_v[ipos]=UNDEF_ROFFINT;
					    p_int_v[ipos]=UNDEF_INT;
					}
				    }
				}
			    }
			}
			fclose(fc);
			return;
		    }
		    if (strcmp(cname,"data")==0 && strcmp(ctype,"int")==0) {
			iok=fread(&n,4,1,fc);
			if (swap==1) SWAP_INT(n);
			for (i=0;i<nx;i++) {
			    for (j=0;j<ny;j++) {
				for (k=0;k<nz;k++) {
				    iok=fread(&aint,4,1,fc);
				    if (swap==1) SWAP_INT(aint);
				    if (storevalue==1) {
					/* map directly to XTGeo form */
					ipos=(nz-(k+1))*ny*nx + j*nx + i;
					p_int_v[ipos]=aint;
					if (p_int_v[ipos]==UNDEF_ROFFINT) {
					    p_int_v[ipos]=UNDEF_INT;
					}
				    }
				}
			    }
			}
			fclose(fc);
			return;
		    }
		}
	    }

	    else{

		/*
		 * The following is just read, not stored
		 */

		xtg_speak(sub,2,"Tag is %s",cname);

		if (strcmp(cname,"eof")==0) break;

		while (1) {
		    _grd3d_roffbinstring(cname, fc);
		    xtg_speak(sub,4,"Reading: %s",cname);
		    if (strcmp(cname,"endtag")==0) break;
		    /* must read other stuff dummy */
		    if (strcmp(cname,"int")==0) {
			_grd3d_roffbinstring(cname,fc);
			iok=fread(&aint,4,1,fc);
			if (swap==1) SWAP_INT(aint);
		    }
		    else if (strcmp(cname,"float")==0) {
			_grd3d_roffbinstring(cname,fc);
			iok=fread(&afloat,4,1,fc);
			if (swap==1) SWAP_FLOAT(afloat);
		    }
		    else if (strcmp(cname,"byte")==0) {
			_grd3d_roffbinstring(cname,fc);
			iok=fread(&abyte,1,1,fc);
		    }
		    else if (strcmp(cname,"char")==0) {
			_grd3d_roffbinstring(cname,fc);
			_grd3d_roffbinstring(cname,fc);
		    }
		    else if (strcmp(cname,"array")==0) {
			_grd3d_roffbinstring(ctype,fc); /* int, float ...*/
			_grd3d_roffbinstring(cname,fc); /* data or? */
			iok=fread(&n,4,1,fc);
			if (swap==1) SWAP_INT(n);
			for (i=0; i<n; i++) {
			    if (strcmp(ctype,"int")==0) {
				iok=fread(&aint,4,1,fc);
				if (swap==1) SWAP_INT(aint);
			    }
			    if (strcmp(ctype,"float")==0) {
				iok=fread(&afloat,4,1,fc);
				if (swap==1) SWAP_FLOAT(afloat);
			    }
			    if (strcmp(ctype,"byte")==0) iok=fread(&abyte,1,1,fc);
			    if (strcmp(ctype,"bool")==0) iok=fread(&abyte,1,1,fc);
			    if (strcmp(ctype,"char")==0) _grd3d_roffbinstring(cname,fc);
			}
		    }
		}
	    }
	}
    }
    fclose(fc);

    if (propstatus<0) {
	xtg_error(sub,"Requested property <%s> not found!",prop_name);
    }

}
