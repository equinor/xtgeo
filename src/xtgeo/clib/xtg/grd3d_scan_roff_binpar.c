/*
 * ############################################################################
 * grd3d_scan_roff_binpar.c
 * Scanning a Roff binary grid for parameter (property) name and type
 * Returns a code for the parameter:
 * 0: parameter is not found
 * 1: Paramater is double
 * 2: Parameter is int
 * Author: JCR
 * ############################################################################
 * $Id: grd3d_scan_par_roff_bin.c,v 1.1 2001/03/14 08:02:29 bg54276 Exp bg54276 $ 
 * $Source: /d/proj/bg/avresmod/src/gplib/GPLExt/RCS/grd3d_scan_par_roff_bin.c,v $ 
 *
 * $Log: grd3d_scan_par_roff_bin.c,v $
 * Revision 1.1  2001/03/14 08:02:29  bg54276
 * Initial revision
 *
 *
 *
 * ############################################################################
 */

/*
 *roff-asc 
 *#ROFF file#
 *#Creator: RMS - Reservoir Modelling System, version 6.0.2#
 *tag filedata 
 *int byteswaptest 1           
 *char filetype  "parameter" *char creationDate  "08/02/2001 21:54:44"
 *endtag 
 *tag version 
 *int major 2           
 *int minor 0           
 *endtag 
 *tag dimensions 
 *int nX 4           
 *int nY 4           
 *int nZ 3           
 *endtag 
 *tag parameter 
 *char name  "Facies"
 *array char codeNames 2           
 * "OLE"
 * "NILS"
 *array int codeValues 2           
 *            1            2
 *array byte data 48          
 *   2   2   1   1   1   1   2   2   1   1   1   1
 *   etc...
 *endtag 
 *tag parameter 
 *char name  "dz"
 *array double data 48          
 *   4.34258423E+01   4.34258118E+01   4.34258423E+01   4.11498108E+01
 *   4.11497192E+01   4.11498108E+01   3.41621094E+01   3.41620789E+01
 *   etc....
 *endtag 
 *tag eof 
 *endtag 
 *
 *
 * OOOPS! ALTERNATIVE:
 *
 *tag parameter 
 *char name  "Facies"
 *char codeNames  "1"
 *int codeValues 1           
 *array byte data 5600        
 


 */

#include <string.h>
#include <stdlib.h>
#include "libxtg.h"
#include "libxtg_.h"

/*
 * ############################################################################
 * Scanning for size... (binary files)
 * ############################################################################
 */

int grd3d_scan_roff_binpar (
			    char    *parname,
			    char    *filename,
			    int     *ndcodes,
			    int     debug
			    )

{
    FILE *fc;
    char cname[ROFFSTRLEN], ctype[ROFFSTRLEN];
    char **cvector;
    char mybyte;
    int  i, iarray[256], myint, iostat;
    int  idum, ncodes, probablyfound, iendian;
    char sub[24]="grd3d_scan_roff_binpar";

    idum=xtgverbose(debug);

    xtg_speak(sub,2,"Entering routine ...");
    /*
     *-------------------------------------------------------------------------
     * Check endiness
     *-------------------------------------------------------------------------
     */
    
    iendian=x_swap_check();
    if (iendian==1) {
	xtg_speak(sub,2,"Machine is little endian (linux intel, windows)");
	x_byteorder(1); /* assumed initially */ 
    }
    else{
	xtg_speak(sub,2,"Machine is big endian (many unix)");
	x_byteorder(0); /* assumed initially */ 
    }





    cvector=calloc(100, sizeof(char *));
    for (i=0; i<100; i++) cvector[i]=calloc(33, sizeof(char));

    *ndcodes=0;

    xtg_speak(sub,2,"Looking for %s",parname);
    /* 
     *-------------------------------------------------------------------------
     * Open file
     *-------------------------------------------------------------------------
     */
  
    xtg_speak(sub,2,"Opening ROFF file: %s",filename);
    fc=fopen(filename,"rb");
    if (fc == NULL) {
	xtg_error(sub,"Cannot open file!");
    }
    xtg_speak(sub,2,"Opening ROFF file...OK!");
    
    /* 
     *=========================================================================
     * Loop file... 
     *=========================================================================
     */

    
    _grd3d_roffbinstring(cname, fc);
    for (idum=1;idum<2000000000;idum++) { /* max 2 giga files */
	
        strcpy(cname,"");
        if ((idum % 10000000)==0) xtg_speak(sub,3,"Working ...");
      
        /* read each byte (char) until "t"=116 is found */
        iostat=fread(&mybyte,1,1,fc);
	if (iostat != 1) xtg_error(sub,"Error (%d) in fread in %s (%d)",iostat,sub,__LINE__);


	if (mybyte == 't') {
	    cname[0]=mybyte;
	    iostat=fread(&mybyte,1,1,fc);
	    if (iostat != 1) xtg_error(sub,"Error (2) in fread in %s",sub);

	    if (mybyte == 'a') {
	      cname[1]=mybyte;
	      iostat=fread(&mybyte,1,1,fc);
	      if (iostat != 1) xtg_error(sub,"Error (3) in fread in %s",sub);
	    }
	    if (mybyte == 'g') {
	      cname[2]=mybyte;	    
	      iostat=fread(&mybyte,1,1,fc);
	      if (iostat != 1) xtg_error(sub,"Error (4) in fread in %s",sub);
	      cname[3]=mybyte;	/* null char? */
	    }    
	}
	else{
	    strcpy(cname,"xxx");
	}
	
      		

	if (strcmp(cname, "tag") == 0) {
	    xtg_speak(sub,3,"Keyword <tag> is found...");
	    _grd3d_roffbinstring(cname, fc);
	    if (strcmp(cname, "tag") == 0) {
		_grd3d_roffbinstring(cname, fc);
		xtg_speak(sub,3,"Tag type is <%s>",cname);
	    }

            /*
             *-----------------------------------------------------------------
             * Getting 'filedata' values
             *-----------------------------------------------------------------
             */
            if (strcmp(cname, "filedata") == 0) {
                xtg_speak(sub,3,"Tag filedata was found");
                myint=_grd3d_getintvalue("byteswaptest",fc);
                xtg_speak(sub,2,"bytewaptest is %d", myint);
		if (myint != 1) {
		    if (iendian==1) x_byteorder(2); 
		    if (iendian==0) x_byteorder(3); 
		    SWAP_INT(mybyte);
		    xtg_speak(sub,1,"Roff file import need swapping");
		    xtg_speak(sub,2,"Bytewaptest is now %d", mybyte);
		    xtg_speak(sub,2,"Byte order flag is now %d", x_byteorder(-1));
		}
	    }


	    /*
	     *-----------------------------------------------------------------
	     * Getting 'parameter' values
	     *-----------------------------------------------------------------
	     */
	    if (strcmp(cname, "eof") == 0) {
		break;
	    }
	    if (strcmp(cname, "parameter") == 0) {
		xtg_speak(sub,3,"Tag parameter was found");
		_grd3d_roffbinstring(cname,fc); /*char*/
		_grd3d_roffbinstring(cname,fc); /*name*/
		_grd3d_roffbinstring(cname,fc); /*actual parameter name*/
		xtg_speak(sub,3,"<%s> is present...",cname);
		xtg_speak(sub,3,"<%s> is wanted...",parname);

		/*
		 * Due to a BUG in RMS IPL RoffExport, the parameter name 
		 * may be "" (empty).
		 * In such cases, it is assumed to be correct!
		 * Must rely on structured programming
		 */
		probablyfound=0;
		if (strcmp(cname,"")==0) {
		    xtg_speak(sub,2,"Empty parameter name ... assuming OK!");
		    probablyfound=1;
		}

		if (probablyfound || strcmp(cname,parname)==0) {
		    xtg_speak(sub,2,"<%s> found!",parname);
		    _grd3d_roffbinstring(cname,fc); /*array or ~nothing*/
		    if (strcmp(cname,"array")==0) {
			_grd3d_roffbinstring(ctype,fc); /*int or double or ...*/
			_grd3d_roffbinstring(cname,fc); /*data or codeNames*/
			if (strcmp(cname,"data")==0) {
			    /* we know the number of items ... return success*/
			    fclose(fc);
			    if (strcmp(ctype,"float")==0) {
				return 1;
			    }
			}
			else if (strcmp(cname,"codeNames")==0) {
			    xtg_speak(sub,3,"codeNames found: ",cname);
			    /* get the number of codes */
			    iostat=fread(&ncodes,4,1,fc);
			    if (iostat != 1) xtg_error(sub,"Error (4) in fread in %s",sub);
			    if (x_byteorder(-1)>1) SWAP_INT(ncodes);
			    xtg_speak(sub,3,"Number of codes: %d",ncodes);
			    *ndcodes=ncodes;
			    /*get the codenames*/
			    _grd3d_getchararray(cvector,ncodes,fc);
			    /* getting array (or scalar)*/
			    _grd3d_roffbinstring(cname,fc); /*array ...*/
			    _grd3d_roffbinstring(ctype,fc); /*int ...*/
			    _grd3d_roffbinstring(cname,fc); /*codeValues*/
			    iostat=fread(&ncodes,4,1,fc);
			    if (iostat != 1) xtg_error(sub,"Error (5) in fread in %s",sub);

			    if (x_byteorder(-1)>1) SWAP_INT(ncodes);
			    if (strcmp(ctype,"int")==0) {
				_grd3d_getintarray(iarray,ncodes,fc);
			    }
			    /* now get to the point */
			    _grd3d_roffbinstring(cname,fc); /*array ...*/
			    _grd3d_roffbinstring(ctype,fc); /*int byte ...*/
			    _grd3d_roffbinstring(cname,fc); /*date*/
			    
			    fclose(fc);
			    if (strcmp(ctype,"int")==0) {
				return 2;
			    }
			    else if (strcmp(ctype,"byte")==0) {
				return 3;
			    }
			}
		    }
		    else{
			/* codeNames etc may NOT be array but scalar */
			_grd3d_roffbinstring(cname,fc); /*codeNames*/
			_grd3d_roffbinstring(cname,fc); /*charvalue*/
			_grd3d_roffbinstring(cname,fc); /*int*/
			_grd3d_roffbinstring(cname,fc); /*codeValues*/
			iostat=fread(&ncodes,4,1,fc);   /*value of codes*/
			if (iostat != 1) xtg_error(sub,"Error (6) in fread in %s",sub);

			if (x_byteorder(-1)>1) SWAP_INT(ncodes);
			_grd3d_roffbinstring(cname,fc); /*array*/
			_grd3d_roffbinstring(ctype,fc); /*int or byte*/
			fclose(fc);
			if (strcmp(ctype,"int")==0) {
			    return 2;
			}
			else if (strcmp(ctype,"byte")==0) {
			    return 3;
			}			
		    }
		}
	    }
	}
    }
    
    /* return 0 if cannot find parname */
    fclose(fc);
    return 0;

}


