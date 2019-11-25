/*
 *******************************************************************************
 *
 * Import ROFF binary property (new implementation sep 2016)
 *
 *******************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"

/*
 *******************************************************************************
 *
 * NAME:
 *    surf_import_irap_bin.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Imports a property on ROFF binary formats. Shall be able to search in
 *    files with multiple records.
 *    Format:
 *------------------------------------------------------------------------------
 *
 * [INTEGER or BYTE or BOOL or CHAR(?):]
 *
 * tag dimensions
 * int nX 99
 * int nY 120
 * int nZ 47
 * endtag
 * tag parameter
 * char name  "Zone"
 * array char codeNames 19
 * ""                                                [no "" in binary format]
 * "Tarbert3B"
 * "Tarbert3A"
 * "Tarbert2B"
 * "Tarbert2A"
 * "Tarbert1C"
 * "Tarbert1B"
 * "Tarbert1A"
 * "Ness3D"
 * ...
 * "SEQ1"
 * array int codeValues 19
 *       0            1            2            3            4            5
 *       6            7            8            9           10           11
 *      12           13           14           15           16           1* 7
 *      18
 * array int data 558360
 *      18           18           18           18           17           17
 *      17           16           16           16           15           15
 *      15           14           14           14           13           13
 *      12           12           11           11           11           10
 *      ...
 *------------------------------------------------------------------------------
 *
 * [FLOAT:]
 * char name  "PORO"
 * array float data 558360
 *  0.00000000E+00   0.00000000E+00   3.34683023E-02   2.26940989E-01
 *  2.51200527E-01   2.42778420E-01   2.46708006E-01   2.03249753E-01
 * ...
 *------------------------------------------------------------------------------
 *
 * ARGUMENTS:
 *    filename       i     File name, character string
 *    scanmode       i     0 for scan, 1 for run
 *    p_type        i/o    Type to be read 1=float, 2=int, 3=byte, ...
 *    p_nx           o     Grid NX (pointer, to return)
 *    p_ny           o     Grid NY aa
 *    p_nz           o     Grid NZ aa
 *    p_ncodes       o     Number of codes, if int/byte
 *    prop_name      i     Name of property to search for
 *    p_int_v        o     Integer array to return (if int mode)
 *    p_double_v     o     Double array to return (if double mode)
 *    p_codenames_v  o     if int: array of chars divided with | -> strings
 *    p_codevalues_v o     if int: array of int codes
 *    option         i     Options flag for later usage
 *    debug          i     Debug level
 *
 * RETURNS:
 *    Function: 0: upon success (parameter OK). If problems <> 0:
 *    -1: parameter not found
 *    Various pointers are updated.
 *
 * TODO/ISSUES/BUGS/NOTES:
 *    Issue: It may be that speed will be enhanced if the x_roffgetfloatarray
 *           etc do a direct mapping to XTGeo variable, including the
 *           IJK order flipping
 *
 * LICENCE:
 *    cf. XTGeo LICENSE
 *******************************************************************************
 */

int grd3d_imp_prop_roffbin (
			    char    *filename,
			    int     scanmode,
			    int     *p_type,
			    int     *p_nx,
			    int     *p_ny,
			    int     *p_nz,
			    int     *p_ncodes,
			    char    *prop_name,
			    int     *p_int_v,
			    double  *p_double_v,
			    char    *p_codenames_v,
			    int     *p_codevalues_v,
			    int     option,
			    int     debug
			    )


{
    int   storevalue;
    int   probablyfound;

    int   i, j, k, ic, n, m, idum, ncodes=1, iok, ntype=1, nn=0;
    int   nx=100, ny=100, nz=100;
    int   swap;
    char  cname[ROFFSTRLEN], ctype[ROFFSTRLEN], csome[ROFFSTRLEN];
    int   propstatus=-1;

    float *p_ftmp_v=NULL;
    int *p_itmp_v=NULL;
    char *p_ctmp_v=NULL;
    unsigned char *p_btmp_v=NULL;

    FILE   *fc=NULL;

    char   s[24]="grd3d_imp_prop_roffbin";

    xtgverbose(debug);
    xtg_speak(s,2,"==== Entering grd3d_import_roff_prop ====");
    xtg_speak(s,2,"Looking for <%s>",prop_name);

    swap=0;
    if (x_byteorder(-1)>1) swap=1;

    *p_ncodes=ncodes; /* initial */

    strcpy(p_codenames_v,"DUMMY");
    xtg_speak(s,2,"Initial (2) codenames string read as: <%s> %d",
              p_codenames_v, &p_codenames_v);


    /*
     *-------------------------------------------------------------------------
     * Open file
     *-------------------------------------------------------------------------
     */

    xtg_speak(s,2,"Opening ROFF file...");
    fc=fopen(filename,"rb");
    if (fc==NULL) {
	xtg_error(s,"Cannot open file!");
    }
    xtg_speak(s,2,"Opening ROFF file...OK!");

    swap=0;
    if (x_byteorder(-1)>1) swap=1;

    /*
     *=========================================================================
     * Loop file...
     *=========================================================================
     */

    x_roffbinstring(cname, fc);
    if (strcmp(cname, "roff-bin") != 0) {
	xtg_error(s,"Not a roff binary file. Stop");
	return(-9);
    }

    for (idum=1;idum<999999;idum++) {

	x_roffbinstring(cname, fc);
	xtg_speak(s,4,"Reading: %s",cname);


	if (strcmp(cname, "endtag") == 0) {
	    xtg_speak(s,2,"Reading: %s",cname);
	}
	else if (strcmp(cname, "tag") == 0) {
	    x_roffbinstring(cname, fc);


	    /*
	     *-----------------------------------------------------------------
	     * Getting 'eof' values
	     *-----------------------------------------------------------------
	     */
	    if (strcmp(cname, "eof") == 0) {
		xtg_speak(s,2,"Tag eof was found");
		goto finally;
	    }
	    /*
	     *-----------------------------------------------------------------
	     * Getting 'dimensions' values
	     *-----------------------------------------------------------------
	     */
	    else if (strcmp(cname, "dimensions") == 0) {
		xtg_speak(s,2,"Tag dimensions was found");
		nx = x_roffgetintvalue("nX", fc);
		ny = x_roffgetintvalue("nY", fc);
		nz = x_roffgetintvalue("nZ", fc);

		*p_nx = nx;
		*p_ny = ny;
		*p_nz = nz;

		xtg_speak(s,2,"Dimensions are %d %d %d", nx, ny, nz);

		xtg_speak(s,2,"Allocate memory for tmp arrays...");
		p_ftmp_v = calloc(nx*ny*nz, sizeof(float));
		p_itmp_v = calloc(nx*ny*nz, sizeof(int));
		p_btmp_v = calloc(nx*ny*nz, sizeof(unsigned char));


	    }

	    /*
	     *-----------------------------------------------------------------
	     * Getting 'parameter' values
	     * Needs reprogramming ... not very elegant ...
	     *-----------------------------------------------------------------
	     */
	    else if (strcmp(cname, "parameter") == 0) {

		xtg_speak(s,2,"Tag parameter was found");

		x_roffbinstring(cname,fc); /*char*/
		x_roffbinstring(cname,fc); /*name*/
		x_roffbinstring(cname,fc); /*actual property name*/

		xtg_speak(s,3,"Property name is <%s>",cname);

		/*
		 * Due to a BUG in RMS IPL RoffExport, the property name
		 * may be "" (empty).
		 * In such cases, it is assumed to be correct!
		 * Must rely on structured programming
		 */

		probablyfound=0;
		if (strcmp(cname,"")==0) probablyfound=1;

		if (strcmp(prop_name,"generic")==0) probablyfound=1;

		if (strcmp(prop_name,"unknown")==0) probablyfound=1;

		if (probablyfound || strcmp(cname,prop_name)==0) {
		    xtg_speak(s,3,"<%s> found!!",cname);
		    storevalue=1;
		    propstatus=0;
		}
		else{
		    storevalue=0;
		}

		if (scanmode == 0) storevalue=0;

	    readarray:

		x_roffbinstring(cname,fc); /*array or...*/
		xtg_speak(s,2,"Reading <%s>",cname);

		if (strcmp(cname,"array")==0) {
		    x_roffbinstring(ctype,fc); /*int or float or ...*/
		    x_roffbinstring(cname,fc); /*data or codeNames or ...?*/

		    /*
		     *----------------------------------------------------------
		     * data:
		     * Note that ROFF use indexing where K runs fastest, from
		     * base!, then J, then I. For XTGeo an Eclipse order
		     * is used, I fastest, then J, then K (from top)
		     * hence the: m=(nz-(k+1))*ny*nx + j*nx + i;
		     *----------------------------------------------------------
		     */


		    if (strcmp(cname,"data")==0) {
			xtg_speak(s,2,"Reading <%s>",cname);
			iok=fread(&n,4,1,fc);
			if (swap==1) SWAP_INT(n);

			if (n != nx*ny*nz) {
			    xtg_error(s,"Error in reading ROFF as "
				      "n != nx*ny*nz. Contact JRIV...");
			}

			if (strcmp(ctype,"float")==0) {
			    x_roffgetfloatarray(p_ftmp_v,n,fc);
			    ntype=1;
			}
			else if (strcmp(ctype,"int")==0) {
			    x_roffgetintarray(p_itmp_v,n,fc);
			    ntype=2;
			}
			else if (strcmp(ctype,"byte")==0) {
			    x_roffgetbytearray(p_btmp_v,n,fc);
			    ntype=3;
			}
			else{
			    xtg_error(s, "Error code 9349");
			}

			xtg_speak(s,2,"PROPTYPE is %d", ntype);

			if (propstatus==0) {
			    *p_type=ntype;
			}

			xtg_speak(s,2,"Store value if %d is 1", storevalue);

			if (storevalue==1) {
			    ic=0;
			    for (i=0;i<nx;i++) {
				for (j=0;j<ny;j++) {
				    for (k=0;k<nz;k++) {

					m=(nz-(k+1))*ny*nx + j*nx + i;

					if (ntype==1) {
					    if (p_ftmp_v[ic]==UNDEF_ROFFFLOAT) {
						p_double_v[m]=UNDEF;
					    }
					    else{
						p_double_v[m]=p_ftmp_v[ic];
					    }
					}
					else if (ntype==2) {
					    if (p_itmp_v[ic]==UNDEF_ROFFINT) {
						p_int_v[m]=UNDEF_INT;
					    }
					    else{
						p_int_v[m]=p_itmp_v[ic];
					    }
					}
					else if (ntype==3) {
					    if (p_btmp_v[ic]==UNDEF_ROFFBYTE) {
						p_int_v[m]=UNDEF_INT;
					    }
					    else{
						/* store as int */
						p_int_v[m]=p_btmp_v[ic];
					    }
					}
					/* char/bool ? etc still missing ...*/
					ic++;
				    }
				}
			    }
			}

			xtg_speak(s,2,"Store value done");

			if (storevalue==1 || (propstatus==0 && scanmode==0)) {
			    if (storevalue == 1) {
				xtg_speak(s,2,"Store value, return with "
					  "code %d", propstatus);
			    }
			    else{
				xtg_speak(s,2,"Scan done, property found OK");
			    }
			    goto finally;
			}

		    }

		    /*
		     *----------------------------------------------------------
		     * codeNames array (and codeValues):
		     *----------------------------------------------------------
		     */

		    else if (strcmp(cname,"codeNames")==0) {
			xtg_speak(s,3,"codeNames found: ",cname);

			/* get the number of codes */
			iok=fread(&ncodes,4,1,fc);
			if (swap==1) SWAP_INT(ncodes);
			xtg_speak(s,3,"Number of codes: %d",ncodes);

			if (propstatus==0 && scanmode==0) {
			    *p_ncodes=ncodes;
			}

			/*get the codenames*/
                        p_ctmp_v = calloc(ncodes * 32, sizeof(char));
			x_roffgetchararray(p_ctmp_v,ncodes,fc);


			//strcpy(p_codenames_v,"SCAN");

			if (storevalue==1) {
                            strcpy(p_codenames_v, p_ctmp_v);
                            xtg_speak(s,3,"Codenames read as: <%s> (ptr: %d)",
                                      p_codenames_v, &p_codenames_v);
			}
			else{
			     xtg_speak(s,3,"Codenames tmp read as: <%s>",
				       p_ctmp_v);
			}

			/* getting values array */
			x_roffbinstring(cname,fc); /*array ...*/
			x_roffbinstring(ctype,fc); /*int ...*/
			x_roffbinstring(cname,fc); /*codeValues*/

			xtg_speak(s,3,"CNAME is <%s>", cname);
			xtg_speak(s,3,"CTYPE is <%s>", ctype);
			iok=fread(&ncodes,4,1,fc);
			if (swap==1) SWAP_INT(ncodes);
			if (strcmp(ctype,"int")==0) {
			    x_roffgetintarray(p_itmp_v,ncodes,fc);
			}
			if (storevalue==1) {
			    for (i=0;i<ncodes;i++) {
				p_codevalues_v[i]=p_itmp_v[i];
                                xtg_speak(s,3,"CODE VALUE: <%d>", p_itmp_v[i]);
			    }
			}

			xtg_speak(s,3,"Codevalues read...");
			goto readarray;
		    }
		}
	    }
	    /*
	     *------------------------------------------------------------------
	     * Getting 'other' tag values (just to scan through)
	     * filedata/version/...
	     *------------------------------------------------------------------
	     */
	    else if (strcmp(cname, "filedata") == 0) {
		xtg_speak(s,2,"Tag filedata was found");
		for (i=0; i<10; i++) {
		    x_roffbinstring(ctype,fc); /*int ...*/
		    if (strcmp(ctype,"int")==0) {
			x_roffbinstring(cname,fc); /* ...xx.. */
			iok=fread(&nn,4,1,fc);
		    }
		    else if (strcmp(ctype,"char")==0) {
			x_roffbinstring(cname,fc); /* ...xx.. */
			x_roffbinstring(csome,fc); /* ...xx.. */
		    }
		    else if (strcmp(ctype,"endtag")==0) {
			xtg_speak(s,2,"Endtag was found");
			break;
		    }
		}
	    }
	    else if (strcmp(cname, "version") == 0) {
		xtg_speak(s,2,"Tag version was found");
		for (i=0; i<10; i++) {
		    x_roffbinstring(ctype,fc); /*int ...*/
		    if (strcmp(ctype,"int")==0) {
			x_roffbinstring(cname,fc); /* ...xx.. */
			iok=fread(&nn,4,1,fc);
		    }
		    else if (strcmp(ctype,"char")==0) {
			x_roffbinstring(cname,fc); /* ...xx.. */
			x_roffbinstring(csome,fc); /* ...xx.. */
		    }
		    else if (strcmp(ctype,"endtag")==0) {
			xtg_speak(s,2,"Endtag was found");
			break;
		    }
		}
	    }
	}
    }


 finally:

    xtg_speak(s,2,"Reached finally mark...");

    free(p_btmp_v);
    free(p_ctmp_v);
    free(p_ftmp_v);
    free(p_itmp_v);

    fclose(fc);

    if (propstatus<0) {
	xtg_speak(s,2,"Requested property <%s> not found!", prop_name);
    }

    return(propstatus);
}
