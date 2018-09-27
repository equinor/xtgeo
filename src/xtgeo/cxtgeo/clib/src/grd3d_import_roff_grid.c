/*
 * ############################################################################
 * grd3d_import_roff_grid
 * Reading a Roff ASCII or binary grid
 * Author: JCR
 *
 * Issue: num_subgr and nnsub is probably the same...
 *
 * ############################################################################
 *
 */


#include <string.h>
#include <stdlib.h>
#include "libxtg.h"
#include "libxtg_.h"

/*
 * ****************************************************************************
 *                       ROFF FILE
 ******************************************************************************
 * The ROFF (Roxar Open File Format) is like this:
 *
 *roff-asc 
 *#ROFF file#
 *#Creator: RMS - Reservoir Modelling System, version 6.0#
 *tag filedata 
 *int byteswaptest 1           
 *char filetype  "grid"
 *char creationDate  "28/03/2000 16:59:16"
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
 *tag translate 
 *float xoffset   4.62994625E+05
 *float yoffset   5.93379900E+06
 *float zoffset  -3.37518921E+01
 *endtag 
 *tag scale 
 *float xscale   1.00000000E+00
 *float yscale   1.00000000E+00
 *float zscale  -1.00000000E+00
 *endtag 
 *tag cornerLines 
 *array float data 150         
 * -7.51105194E+01  -4.10773730E+03  -1.86212000E+03  -7.51105194E+01
 * -4.10773730E+03  -1.72856909E+03  -8.36509094E+02  -2.74306006E+03
 *  ....
 *endtag 
 *tag zvalues 
 *array byte splitEnz 100         
 *  1   1   1   1   1   1   4   4   1   1   1   1
 * ....
 *endtag
 *tag active 
 *array bool data 48          
 *  1   1   1   1   1   1   1   1   1   1   1   1
 *  1   1   1   1   1   1   1   1   1   1   1   1
 *  1   1   1   1   1   1   1   1   1   1   1   1
 *  1   1   1   1   1   1   1   1   1   1   1   1
 *endtag 
 *... ETC
 * ----------------------------------------------------------------------------
 *
 */   


void grd3d_import_roff_grid (
			     int     *num_act,
			     int     *num_subgrds,
			     double  *p_coord_v,
			     double  *p_zcorn_v,
			     int     *p_actnum_v,
			     int     *p_subgrd_v,
			     int     nnsubs,
			     char    *filename,
			     int     debug
			     )


{
    char cname[9];
    FILE *fc;
    char sub[24]="grd3d_import_roff_grid";

    xtgverbose(debug);

    xtg_speak(sub,2,"==== Entering grd3d_import_roff_grid ====");
    /* 
     *-------------------------------------------------------------------------
     * Open file
     *-------------------------------------------------------------------------
     */
  
    xtg_speak(sub,2,"Opening ROFF file...");
    fc=fopen(filename,"rb");
    if (fc == NULL) {
	xtg_error(sub,"Cannot open file!");
    }
    xtg_speak(sub,2,"Opening ROFF file...OK!");
    
    
    /* 
     *=========================================================================
     * Loop file... It is NOT necessary to do many tests; that should be done
     * by the calling PERL script?
     *=========================================================================
     */
     
    x_fread(cname,7,1,fc,__FILE__,__LINE__);
    fclose(fc);
    cname[7]='\0';
    xtg_speak(sub,2,"Header is %s\n", cname);

    if (strcmp(cname,"roff-as")==0) {
	xtg_speak(sub,2,"ROFF ASCII file...");
	_grd3d_imp_roff_asc_grd(
			       num_act,
			       num_subgrds,
			       p_coord_v,
			       p_zcorn_v,
			       p_actnum_v,
			       p_subgrd_v,
			       nnsubs,
			       filename,
			       debug
			       );
    }
    else{
	xtg_speak(sub,2,"ROFF Binary file...");
	_grd3d_imp_roff_bin_grd(
				num_act,
				num_subgrds,
				p_coord_v,
				p_zcorn_v,
				p_actnum_v,
				p_subgrd_v,
				nnsubs,
				filename,
				debug
				);

    }




}


/*
 * ############################################################################
 * Importing ASCII version
 * ############################################################################
 */

void _grd3d_imp_roff_asc_grd (
			     int     *num_act,
			     int     *num_subgrds,
			     double  *p_coord_v,
			     double  *p_zcorn_v,
			     int     *p_actnum_v,
			     int     *p_subgrd_v,
			     int     nnsubs,
			     char    *filename,
			     int     debug
			     )


{
    
    int    iok, line;
    int    i, j, k, ipos, num_cornerlines, num_active, num_zdata, num_splitenz;
    int    nx, ny, nz, ivalue, nsub;
    char   *splitenz_v=NULL;
    float  xoffset, yoffset, zoffset, xscale, yscale, zscale, fvalue;
    float  *cornerlines_v=NULL;
    float  *zdata_v=NULL;
    char   cline[133], ctype[ROFFSTRLEN], carray[ROFFSTRLEN];
    char   cname[ROFFSTRLEN], cx[ROFFSTRLEN];
    FILE   *fc;
    char sub[24]="_grd3d_imp_roff_asc_grd";
    
    xtgverbose(debug);

    xtg_speak(sub,2,"==== Entering _grd3d_imp_roff_asc_grd ====");
    /* 
     *-------------------------------------------------------------------------
     * Open file
     *-------------------------------------------------------------------------
     */
  
    xtg_speak(sub,2,"Opening ROFF file...");
    fc=fopen(filename,"rb");
    if (fc == NULL) {
	xtg_error(sub,"Cannot open file!");
    }
    xtg_speak(sub,2,"Opening ROFF file...OK!");
    

    /* 
     *=========================================================================
     * Loop file... It is NOT necessary to do many tests; that should be done
     * by the calling PERL script?
     *=========================================================================
     */
     

    for (line=1;line<9999999;line++) {
	
	/* Get offsets */
	x_fgets(cline,132,fc);
      
	/*
	 *---------------------------------------------------------------------
	 * Getting 'translate' values
	 *---------------------------------------------------------------------
	 */
	if (strncmp(cline, "tag translate", 13) == 0) {
	    xtg_speak(sub,3,"Tag translate was found");
	    x_fgets(cline,132,fc);
	    iok=sscanf(cline,"%s %s %f",ctype, cname, &xoffset);
	    if (iok==0) xtg_error(sub,"IOK is zero!");
	    xtg_speak(sub,3,"xoffset is %f", xoffset);
	    x_fgets(cline,132,fc);
	    iok=sscanf(cline,"%s %s %f",ctype, cname, &yoffset);
	    xtg_speak(sub,3,"yoffset is %f", yoffset);
	    x_fgets(cline,132,fc);
	    iok=sscanf(cline,"%s %s %f",ctype, cname, &zoffset);
	    xtg_speak(sub,3,"zoffset is %f", zoffset);
	    x_fgets(cline,132,fc); /* endtag */
	}
	if (strncmp(cline,"char filetype",13)==0) {
	    iok=sscanf(cline,"%s %s %s", ctype, cname, cx);
	    if (strcmp(cx,"\"grid\"")==0) {
		xtg_speak(sub,3,"Filetype is ROFF grid ... OK");
	    }
	    else{
		xtg_error(sub,"Filetype is wrong: %s",cx);
		return;
	    }
	}
	/*
	 *---------------------------------------------------------------------
	 * Getting 'scale' values
	 *---------------------------------------------------------------------
	 */
	if (strncmp(cline, "tag scale", 9) == 0) {
	    xtg_speak(sub,3,"Tag scale was found");
	    x_fgets(cline,132,fc);
	    iok=sscanf(cline,"%s %s %f",ctype, cname, &xscale);
	    xtg_speak(sub,3,"xscale is: %f", xscale);
	    x_fgets(cline,132,fc);
	    iok=sscanf(cline,"%s %s %f",ctype, cname, &yscale);
	    xtg_speak(sub,3,"zscale is: %f", yscale);
	    x_fgets(cline,132,fc);
	    iok=sscanf(cline,"%s %s %f",ctype, cname, &zscale);
	    xtg_speak(sub,3,"zscale is: %f", zscale);
	    x_fgets(cline,132,fc); /* endtag */
	}

	/*
	 *---------------------------------------------------------------------
	 * Getting 'dimensions' values
	 *---------------------------------------------------------------------
	 */
	if (strncmp(cline, "tag dimensions", 14) == 0) {
	    xtg_speak(sub,3,"Tag dimensions was found");
	    x_fgets(cline,132,fc);
	    iok=sscanf(cline,"%s %s %d",ctype, cname, &nx);
	    xtg_speak(sub,3,"NX is %d", nx);
	    x_fgets(cline,132,fc);
	    iok=sscanf(cline,"%s %s %d",ctype, cname, &ny);
	    xtg_speak(sub,3,"NY is %d", ny);
	    x_fgets(cline,132,fc);
	    iok=sscanf(cline,"%s %s %d",ctype, cname, &nz);
	    xtg_speak(sub,3,"NZ is %d", nz);
	    x_fgets(cline,132,fc); /* endtag */

	    /* assign subgrid values */
	    *num_subgrds=1; /* no subgrids (i.e. all is subgrd 1) initially */
	    p_subgrd_v[0]=nz;

	    /*
	     * The ACTNUM (Active) codeword may not be present if all cells are
	     * active. Must therefore initialize to 1.
	     *
	     */
	    
	    for (i=0; i<nx*ny*nz; i++) {
		p_actnum_v[i]=1;
	    }

	}


	/*
	 *---------------------------------------------------------------------
	 * Getting 'subgrids' values
	 * Note that subgrids are counted from top and downwards. I.e. in
	 * and right-handed ROFF file with K counting from bottom, the
	 * subgrids are counted from K=nz and down!
	 * Ther is also a quirk with subgrids as one single zone may have
	 * one subgrid... If nnsubs=1, then this post will be skipped
	 *---------------------------------------------------------------------
	 */
	if (strncmp(cline, "tag subgrids", 12) == 0 && nnsubs>1) {
	    xtg_speak(sub,3,"Tag subgrids was found");
	    x_fgets(cline,132,fc);

	    iok=sscanf(cline,"%s %s %s %d", carray, ctype, cname, &nsub);
	    xtg_speak(sub,2,"Number of subgrids are are %d", nsub);
	    *num_subgrds=nsub;
	    xtg_speak(sub,2,"Reading <subgrids>...");
	    for (i=0; i<nsub; i++) {
		iok=fscanf(fc,"%d",&ivalue);
		p_subgrd_v[i]=ivalue;
	    }
	    xtg_speak(sub,2,"Reading <subgrids>...DONE!");
	    x_fgets(cline,132,fc); /* endtag */
	}

	/*
	 *---------------------------------------------------------------------
	 * Getting 'corneLines' array
	 *---------------------------------------------------------------------
	 */

	if (strncmp(cline, "tag cornerLines", 15) == 0) {
	    xtg_speak(sub,3,"Tag cornerLines was found");
	    x_fgets(cline,132,fc);
	    iok=sscanf(cline,"%s %s %s %d",carray, ctype, cname, 
		       &num_cornerlines);
	    /* 
	     * Allocating space
	     */
	    xtg_speak(sub,2,"Allocating memory for corneLines: %d of type %s",
		    num_cornerlines,ctype);
	    cornerlines_v=calloc(num_cornerlines,4);
	    xtg_speak(sub,2,"Allocating memory ... DONE!");

	    /*
	     * Read data values
	     */
	    xtg_speak(sub,2,"Reading <cornerLines> ...");
	    for (i=0; i<num_cornerlines; i++) {
		iok=fscanf(fc,"%f",&fvalue);
		if (fvalue == 9999900.0000) {
		    fvalue=-9999.99;
		}
		cornerlines_v[i]=fvalue;
		if (debug >= 4) xtg_speak(sub,4,"CornerLine [%d] %f",i,fvalue);
	    }
	    x_fgets(cline,132,fc); /* endtag */
	    xtg_speak(sub,2,"Reading <cornerLines> ...DONE!");
	}


	/*
	 *---------------------------------------------------------------------
	 * Getting 'zvalues' arrays
	 *---------------------------------------------------------------------
	 */

	if (strncmp(cline, "tag zvalues", 11) == 0) {
	    xtg_speak(sub,3,"Tag zvalues was found");
	    x_fgets(cline,132,fc);
	    iok=sscanf(cline,"%s %s %s %d",carray, ctype, cname, &num_splitenz);
	    /*
	     * Read split values
	     */ 
 	    xtg_speak(sub,2,"Allocating memory for splitenz: %d of type %s",num_splitenz,ctype);
	    splitenz_v=calloc(num_splitenz,1);
	    xtg_speak(sub,2,"Allocating memory ... DONE!");
	    xtg_speak(sub,2,"Reading <splitEnz>...");
	    for (i=0; i<num_splitenz; i++) {
		iok=fscanf(fc,"%d",&ivalue);
		splitenz_v[i]=ivalue;
	    }
	    x_fgets(cline,132,fc); /* endtag */
	    xtg_speak(sub,2,"Reading <splitEnz>...OK!");
	    /*
	     * Read data values
	     */      
	    x_fgets(cline,132,fc);
	    iok=sscanf(cline,"%s %s %s %d",carray, ctype, cname, &num_zdata);
	    xtg_speak(sub,2,"Reading <data>...");
	    xtg_speak(sub,2,"Allocating memory for zdata: %d of type %s",num_zdata,ctype);
	    zdata_v=calloc(num_zdata,4);
	    xtg_speak(sub,2,"Allocating memory ... DONE!");

	    for (i=0; i<num_zdata; i++) {
		iok=fscanf(fc,"%f",&fvalue);
		zdata_v[i]=fvalue;
	    }
	    x_fgets(cline,132,fc); /* endtag */
	    xtg_speak(sub,2,"Reading <data>...OK!");
	}

	/*
	 *---------------------------------------------------------------------
	 * Getting 'active' array
	 *---------------------------------------------------------------------
	 */
	if (strncmp(cline, "tag active", 10) == 0) {
	    xtg_speak(sub,3,"Tag active was found");
	    x_fgets(cline,132,fc);
	    iok=sscanf(cline,"%s %s %s %d",carray, ctype, cname, &num_active);
	    /*
	     * Read values
	     */

	    xtg_speak(sub,2,"Reading <active>...");
	    if (strncmp(ctype,"bool",4) == 0) {strcpy(ctype,"byte");}
	    xtg_speak(sub,2,"There should be a total of %d cells", num_active);

	    if (num_active != (nx*ny*nz)) {
		xtg_error(sub,"Error! num_active: %d  NOT EQUAL nx*ny*nz: %d",
			num_active, nx*ny*nz);
	    }

	    for (i=0;i<nx;i++) {
		for (j=0;j<ny;j++) {
		    for (k=0;k<nz;k++) {			    
			iok=fscanf(fc,"%d",&ivalue);
			/* map directly to XTG form */
			ipos=(nz-(k+1))*ny*nx + j*nx + i;
			p_actnum_v[ipos]=ivalue;
			    
		    }
		}
	    }

	    x_fgets(cline,132,fc); /* endtag */
	    xtg_speak(sub,2,"Reading <active>...OK!");
	}
	xtg_speak(sub,4,"Reading line:\n   %s", cline);

	if (strncmp(cline, "tag eof", 7) == 0) {
	    xtg_speak(sub,2,"End of file detected");
	    break;
	}

	if (strcmp(cline, "tag parameter") == 0) {
	    xtg_speak(sub,2,"End of grid description detected");
	    break;
	}
    }
    fclose(fc);

    _grd3d_roff_to_xtg_grid(
			    nx,
			    ny,
			    nz,
			    xoffset,
			    yoffset,
			    zoffset,
			    xscale,
			    yscale,
			    zscale,
			    cornerlines_v,
			    splitenz_v,
			    zdata_v,
			    num_act,
			    num_subgrds,
			    p_coord_v,
			    p_zcorn_v,
			    p_actnum_v,
			    p_subgrd_v,
			    debug
			    );

    xtg_speak(sub,2,"Unallocating memory for tmp arrays ...");
    xtg_speak(sub,3,"... cornerlines_v:");
    free(cornerlines_v);
    xtg_speak(sub,3,"... splitenz_v:");
    free(splitenz_v);
    xtg_speak(sub,3,"... zdata_v:");
    free(zdata_v);
    xtg_speak(sub,3,"Unallocating memory for tmp arrays ... DONE!");
    xtg_speak(sub,2,"==== Exiting _grd3d_imp_roff_asc_grd ====");


}

/*
 * ############################################################################
 * Reading a Roff BINARY grid
 * Author: JCR
 * ############################################################################
 */

void _grd3d_imp_roff_bin_grd (
			      int     *num_act,
			      int     *num_subgrds,
			      double  *p_coord_v,
			      double  *p_zcorn_v,
			      int     *p_actnum_v,
			      int     *p_subgrd_v,
			      int     nnsub,
			      char    *filename,
			      int     debug
			      )


{
    FILE   *fc;
    char   cname[ROFFSTRLEN];
    char   mybyte;
    int    i, j, k, ipos, idum;
    int    num, iendiness;
    float  xoffset=0.0, yoffset=0.0, zoffset=0.0;
    float  xscale=1.0, yscale=1.0, zscale=1.0;
    float  *cornerlines_v=NULL, *zdata_v=NULL;
    char   *splitenz_v=NULL;
    int    nx=1, ny=1, nz=1, nnact=0;
    char   sub[24]="_grd3d_imp_roff_bin_grd";

    xtgverbose(debug);

    xtg_speak(sub,2,"==== Entering _grd3d_imp_roff_bin_grd ====");
    /* 
     *-------------------------------------------------------------------------
     * Check endiness
     *-------------------------------------------------------------------------
     */
    
    iendiness=x_swap_check();
    if (iendiness==1) {
	xtg_speak(sub,2,"Machine is little endian.");
    }
    else{
	xtg_speak(sub,2,"Machine is big endian");
    }

    /* 
     *-------------------------------------------------------------------------
     * Open file
     *-------------------------------------------------------------------------
     */
  
    xtg_speak(sub,2,"NNSUB value is %d",nnsub);

    xtg_speak(sub,2,"Opening ROFF file...");
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
    for (idum=1;idum<99999;idum++) {
	
	_grd3d_roffbinstring(cname, fc);
      
	if (strcmp(cname, "tag") == 0) {
	    _grd3d_roffbinstring(cname, fc);


	    /*
	     *-----------------------------------------------------------------
	     * Getting 'translate' values
	     *-----------------------------------------------------------------
	     */
	    if (strcmp(cname, "filedata") == 0) {
		xtg_speak(sub,2,"Tag filedata...");
		idum=_grd3d_getintvalue("byteswaptest",fc);
		_grd3d_roffbinstring(cname, fc); /* char */
		_grd3d_roffbinstring(cname, fc); /* filetype */
		_grd3d_roffbinstring(cname, fc); /* grid or parameter */
		if (strcmp(cname,"grid")==0) {
		    xtg_speak(sub,3,"Type is ROFF grid ... OK");
		}
		else{
		    xtg_error(sub,"Not a ROFF grid file!");
		}
	    }
	    if (strcmp(cname, "translate") == 0) {
		xtg_speak(sub,3,"Tag translate was found");
		xoffset=_grd3d_getfloatvalue("xoffset",fc);
		xtg_speak(sub,3,"xoffset is %f", xoffset);
		yoffset=_grd3d_getfloatvalue("yoffset",fc);
		xtg_speak(sub,3,"yoffset is %f", yoffset);
		zoffset=_grd3d_getfloatvalue("zoffset",fc);
		xtg_speak(sub,3,"zoffset is %f", zoffset);
	    }
	    /*
	     *-----------------------------------------------------------------
	     * Getting 'scale' values
	     *-----------------------------------------------------------------
	     */
	    if (strcmp(cname, "scale") == 0) {
		xtg_speak(sub,3,"Tag scale was found");
		xscale=_grd3d_getfloatvalue("xscale",fc);
		xtg_speak(sub,3,"xscale is %f", xscale);
		yscale=_grd3d_getfloatvalue("yscale",fc);
		xtg_speak(sub,3,"yscale is %f", yscale);
		zscale=_grd3d_getfloatvalue("zscale",fc);
		xtg_speak(sub,3,"zscale is %f", zscale);
	    }
	    /*
	     *-----------------------------------------------------------------
	     * Getting 'dimensions' values
	     *-----------------------------------------------------------------
	     */
	    if (strcmp(cname, "dimensions") == 0) {
		xtg_speak(sub,3,"Tag dimensions was found");
		nx=_grd3d_getintvalue("nX",fc);
		xtg_speak(sub,3,"nX is %d", nx);
		ny=_grd3d_getintvalue("nY",fc);
		xtg_speak(sub,3,"nY is %d", ny);
		nz=_grd3d_getintvalue("nZ",fc);
		xtg_speak(sub,3,"nZ is %d", nz);
	    }
	    /*
	    if (nx <= 0 || ny <=0 || nz <=0) {
		xtg_error(sub,"Dimension error; values <=0 for n(xyz). STOP!");
	    }
	    */
	    /*
	     *-----------------------------------------------------------------
	     * Getting 'subgrids' array
	     *-----------------------------------------------------------------
	     */
	    if (strcmp(cname, "subgrids") == 0) {
		xtg_speak(sub,3,"Tag subgrids was found");
		if (nnsub > 1) {
		    num=_grd3d_getintvalue("array",fc);
		    xtg_speak(sub,2,"Number of subgrids are are %d", num);
		    *num_subgrds=num;
		    _grd3d_getintarray(p_subgrd_v,num,fc);
		}
		else{
		    xtg_speak(sub,2,"Number of subgrids are are %d", nnsub);
		    num=_grd3d_getintvalue("nLayers",fc);
		    num=nnsub;
		    *num_subgrds=num;
		    xtg_speak(sub,3,"Number of subgrids are are %d", num);
		}
	    }
	    /*
	     *-----------------------------------------------------------------
	     * Getting 'cornerLines' array
	     *-----------------------------------------------------------------
	     */
	    if (strcmp(cname, "cornerLines") == 0) {
		xtg_speak(sub,3,"Tag cornerLines was found");
		num=_grd3d_getintvalue("array",fc);
		xtg_speak(sub,3,"Allocating cornerLines: %d",num);
		cornerlines_v=calloc(num,4);
		_grd3d_getfloatarray(cornerlines_v,num,fc);
		if (debug>=4) {
		    for (i=0;i<10;i++) {
			xtg_speak(sub,4,"Cornerline no. %d is %f",i,cornerlines_v[i]);
		    }
		}

	    }
	    /*
	     *-----------------------------------------------------------------
	     * Getting 'zvalues' array
	     *-----------------------------------------------------------------
	     */
	    if (strcmp(cname, "zvalues") == 0) {
		xtg_speak(sub,3,"Tag zvalues was found");
		num=_grd3d_getintvalue("array",fc);
		xtg_speak(sub,3,"Allocating zvalues: %d",num);
		splitenz_v=calloc(num,1);
		_grd3d_getbytearray(splitenz_v,num,fc);
		if (debug>=4) {
		    for (i=0;i<10;i++) {
			xtg_speak(sub,4,"SplitEnz no. %d is %d",i,splitenz_v[i]);
		    }
		}
		num=_grd3d_getintvalue("array",fc);
		xtg_speak(sub,3,"Allocating num: %d",num);
		zdata_v=calloc(num,4);
		_grd3d_getfloatarray(zdata_v,num,fc);
		if (debug>=4) {
		    for (i=0;i<10;i++) {
			xtg_speak(sub,4,"Zdata no. %d is %f",i,zdata_v[i]);
		    }
		}
	    }
	    /*
	     *-----------------------------------------------------------------
	     * Getting 'active' array
	     *-----------------------------------------------------------------
	     */
	    if (strcmp(cname, "active") == 0) {
		nnact = 0;
		xtg_speak(sub,3,"Tag active was found");
		num=_grd3d_getintvalue("array",fc);
		
		for (i=0;i<nx;i++) {
		    for (j=0;j<ny;j++) {
			for (k=0;k<nz;k++) {			    
			    x_fread(&mybyte,1,1,fc,__FILE__,__LINE__);
			    /* map directly to XTG form */
			    ipos=(nz-(k+1))*ny*nx + j*nx + i;
			    p_actnum_v[ipos]=mybyte;
			    if (p_actnum_v[ipos]==1) nnact++;
			}
		    }
		}
		*num_act = nnact;
	    }
	}




    }
    
    
    xtg_speak(sub,2,"Number of active cells is %d",*num_act);


    fclose(fc);


    _grd3d_roff_to_xtg_grid(
			    nx,
			    ny,
			    nz,
			    xoffset,
			    yoffset,
			    zoffset,
			    xscale,
			    yscale,
			    zscale,
			    cornerlines_v,
			    splitenz_v,
			    zdata_v,
			    num_act,
			    num_subgrds,
			    p_coord_v,
			    p_zcorn_v,
			    p_actnum_v,
			    p_subgrd_v,
			    debug
			    );



    xtg_speak(sub,2,"Unallocating memory for tmp arrays ...");
    free(cornerlines_v);
    free(splitenz_v);
    free(zdata_v);
    xtg_speak(sub,3,"Unallocating memory for tmp arrays ... DONE!");


}


/*
 *############################################################################
 *
 * Reading a string in binary file terminated with NULL
 * Something is strange here; see result if x_fread are used...
 *############################################################################
 */

int _grd3d_roffbinstring(char *bla, FILE *fc) 

{
    int i, ier;
    char mybyte;
    
    for (i=0;i<ROFFSTRLEN;i++) {
        /* x_fread(&mybyte,1,1,fc,__FILE__,__LINE__); */
        ier=fread(&mybyte,1,1,fc);
	bla[i]=mybyte;
        if (mybyte==0) break;
    }
    
    return 1;
}
 


float _grd3d_getfloatvalue(char *name, FILE *fc) 
{
    char bla[ROFFSTRLEN];
    float myfloat;

    _grd3d_roffbinstring(bla, fc);
    if (strcmp(bla,"float")==0) {
	_grd3d_roffbinstring(bla, fc);
	if (strcmp(bla,name)==0) {
	    x_fread(&myfloat,4,1,fc,__FILE__,__LINE__);
	    if (x_byteorder(-1)>1) SWAP_FLOAT(myfloat);
	    return myfloat;
	}
    }
    return -1.0;
}


int _grd3d_getintvalue(char *name, FILE *fc) 
{
    char bla[ROFFSTRLEN];
    int  myint;

    if (strcmp(name,"array")==0) {

	/* return -1 in case not a array as suggested... */
	_grd3d_roffbinstring(bla, fc); /* array */
	if (strcmp(bla,"array")!=0) {
	    return -1;
	}
	_grd3d_roffbinstring(bla, fc); /* int or float */
	_grd3d_roffbinstring(bla, fc); /* data */
	x_fread(&myint,4,1,fc,__FILE__,__LINE__);
	if (x_byteorder(-1)>1) SWAP_INT(myint);
	return myint;
    }
    else{
	_grd3d_roffbinstring(bla, fc);
	if (strcmp(bla,"int")==0) {
	    _grd3d_roffbinstring(bla, fc);
	    if (strcmp(bla,name)==0) {
		x_fread(&myint,4,1,fc,__FILE__,__LINE__);
		if (x_byteorder(-1)>1) SWAP_INT(myint);
		return myint;
	    }
	}
    }
    return -1;
}

void _grd3d_getfloatarray(float *array, int num, FILE *fc) 
{
    float afloat;
    int   i;

    for (i=0;i<num;i++) {
	x_fread(&afloat,4,1,fc,__FILE__,__LINE__);
	if (x_byteorder(-1)>1) SWAP_FLOAT(afloat);
	array[i]=afloat;
    }


    /* PREVIOUS WAY: fread(array,4,num,fc); */

	


}

void _grd3d_getbytearray(char *array, int num, FILE *fc) 
{
    int  i;
    char abyte;

    for (i=0;i<num;i++) {
	x_fread(&abyte,1,1,fc,__FILE__,__LINE__);
	array[i]=abyte;
    }
}

void _grd3d_getintarray(int *array, int num, FILE *fc) 
{
    int  i;
    int aint;

    for (i=0;i<num;i++) {
	x_fread(&aint,4,1,fc,__FILE__,__LINE__);
	if (x_byteorder(-1)>1) SWAP_SHORT(aint);
	array[i]=aint;
    }
}

void _grd3d_getchararray(char **array, int num, FILE *fc) 
{
    int  i, j;
    char c[ROFFSTRLEN];
    char sub[24]="_grd3d_getchararray";

    for (i=0;i<num;i++) {
	_grd3d_roffbinstring(c,fc);
	xtg_speak(sub,4,"Reading: <%s>\n",c);
	for (j=0;j<100;j++) {
	    array[i][j]=c[j];
	    if (c[j] == '\0') break;
	}
    }    
}


/*
 *############################################################################
 *
 * The next big issue is to get these data on right XTG format
 *
 *############################################################################
 */


void _grd3d_roff_to_xtg_grid (
			      int     nx,
			      int     ny,
			      int     nz,
			      float   xoffset,
			      float   yoffset,
			      float   zoffset,
			      float   xscale,
			      float   yscale,
			      float   zscale,
			      float   *cornerlines_v,
			      char    *splitenz_v,
			      float   *zdata_v,
			      int     *num_act,
			      int     *num_subgrds,
			      double  *p_coord_v,
			      double  *p_zcorn_v,
			      int     *p_actnum_v,
			      int     *p_subgrd_v,
			      int     debug
			      )
    
{

    int    i, j, k, ij, iuse, icnt;
    int    l, isplit, ipos;
    int    ib, ic, niijjkk, nipjjkk, niijpkk, niijjkp; 
    int	   nipjpkk, nipjjkp, niijpkp, nipjpkp;
    int    *lookup_v;
    int    nxyz;
    double z, *z_nw_v, *z_se_v, *z_sw_v, *z_ne_v; 
    double *xp_bot_v, *yp_bot_v, *zp_bot_v, *xp_top_v, *yp_top_v, *zp_top_v;
    double zz[9];
    char   sub[24]="_grd3d_roff_to_xtg_grid";

    xtgverbose(debug);


    xtg_speak(sub,2,"Transforming to internal XTG representation ...");

    xtg_speak(sub,1,"NX NY NZ %d %d %d: ",nx, ny, nz);


    /*
     *---------------------------------------------------------------------
     * Getting corners of each i,j pillar
     *---------------------------------------------------------------------
     */

    xtg_speak(sub,3,"Extracting pillar tops and bottoms...");
    xp_bot_v = calloc((nx+1)*(ny+1), sizeof(double));
    yp_bot_v = calloc((nx+1)*(ny+1), sizeof(double));
    zp_bot_v = calloc((nx+1)*(ny+1), sizeof(double));
    xp_top_v = calloc((nx+1)*(ny+1), sizeof(double));
    yp_top_v = calloc((nx+1)*(ny+1), sizeof(double));
    zp_top_v = calloc((nx+1)*(ny+1), sizeof(double));


    /* Note that counting is j faster than i */
    
    for (i=0;i<=nx;i++) {
	for (j=0;j<=ny;j++) {
	    ipos=6*(i*(ny+1)+j);
	    ij=(i*(ny+1)+j);
  	    xp_bot_v[ij]=cornerlines_v[ipos];
  	    yp_bot_v[ij]=cornerlines_v[ipos+1];
  	    zp_bot_v[ij]=cornerlines_v[ipos+2];
  	    xp_top_v[ij]=cornerlines_v[ipos+3];
  	    yp_top_v[ij]=cornerlines_v[ipos+4];
  	    zp_top_v[ij]=cornerlines_v[ipos+5];


	    if (debug >= 4) {
		xtg_speak(sub,4,"Pillar info:");
		xtg_speak(sub,4,"====> I, J (0 offset!)  %d  %d",i, j);
		xtg_speak(sub,4,"BOT X Y Z:  %f %f %f",
			xp_bot_v[ij], yp_bot_v[ij], zp_bot_v[ij]);
		xtg_speak(sub,4,"TOP X Y Z:  %f %f %f",
			xp_top_v[ij], yp_top_v[ij], zp_top_v[ij]);

	    }

	    xp_bot_v[ij]=(xp_bot_v[ij]+xoffset)*xscale; 
	    yp_bot_v[ij]=(yp_bot_v[ij]+yoffset)*yscale; 
	    zp_bot_v[ij]=(zp_bot_v[ij]+zoffset)*zscale; 
	    xp_top_v[ij]=(xp_top_v[ij]+xoffset)*xscale; 
	    yp_top_v[ij]=(yp_top_v[ij]+yoffset)*yscale; 
	    zp_top_v[ij]=(zp_top_v[ij]+zoffset)*zscale; 

	    if (debug >= 4) {
		xtg_speak(sub,4,"Pillar info TRANSLATED:");
		xtg_speak(sub,4,"====> I, J (0 offset!)  %d  %d",i, j);
		xtg_speak(sub,4,"BOT X Y Z:  %f %f %f",
			xp_bot_v[ij], yp_bot_v[ij], zp_bot_v[ij]);
		xtg_speak(sub,4,"TOP X Y Z:  %f %f %f",
			xp_top_v[ij], yp_top_v[ij], zp_top_v[ij]);

	    }
	}   
    }
    xtg_speak(sub,3,"Extracting pillar tops and bottoms...OK!");
    
    /*
     *---------------------------------------------------------------------
     * Getting corners of each i,j pillar
     *---------------------------------------------------------------------
     */

    xtg_speak(sub,3,"Make splitnode lookup table...");
    nxyz=(nx+1)*(ny+1)*(nz+1);
    if ((lookup_v = calloc(nxyz+2, sizeof(int))) != NULL) {
	xtg_speak(sub,3,"Allocating lookup_v");
    }
    else{
	xtg_warn(sub,1,"NXYZ is %d NX NY NZ %d %d %d",nxyz, nx, ny, nz);
	xtg_error(sub,"Allocating lookup_v FAILED");
    }	
    lookup_v[0]=0;
    for (i=0; i<nxyz; i++) {
	lookup_v[i+1]=lookup_v[i] + splitenz_v[i];
    }
    xtg_speak(sub,3,"Make splitnode lookup table...OK!");
	

    /* xtg_speak(sub,3,"Make z pr 3D node..."); */
    z_sw_v=calloc(8, sizeof(double));
    z_se_v=calloc(8, sizeof(double));
    z_nw_v=calloc(8, sizeof(double));
    z_ne_v=calloc(8, sizeof(double));
    

    xtg_speak(sub,4,"Creating arrays on XTG form...");

    xtg_speak(sub,3,"--> Grid array on XTG form...");

    /*
     *---------------------------------------------------------------------
     * Making arrays on internal grid format (Eclipse like). This
     * is 8 corners with X,Y,Z, and ordered in a way such that I is
     * cycling fastest
     *---------------------------------------------------------------------
     */

    /* COORD lines, coutning i fater than j*/
    ib=0;
    xtg_speak(sub,2,"Ordering COORDs ...");
    for (j=0;j<=ny;j++) {
	for (i=0;i<=nx;i++) {
	    ij=(i*(ny+1)+j); /* ij is counted from 0 and j fastest */
	    p_coord_v[ib++]=xp_top_v[ij];
	    p_coord_v[ib++]=yp_top_v[ij];
	    p_coord_v[ib++]=zp_top_v[ij];
	    p_coord_v[ib++]=xp_bot_v[ij];
	    p_coord_v[ib++]=yp_bot_v[ij];
	    p_coord_v[ib++]=zp_bot_v[ij];
	}
    }
    xtg_speak(sub,2,"Ordering COORDs ...DONE!");


    ib=0;
    for (l=(nz-1);l>=-1;l--) {
	xtg_speak(sub,3,"Working with layer: %d",l);
	for (j=0;j<ny;j++) {
	    for (i=0;i<nx;i++) {

		k=l;
		if (l==-1) k=0;


		niijjkk=i*(ny+1)*(nz+1)+j*(nz+1)+k;
		nipjjkk=(i+1)*(ny+1)*(nz+1)+j*(nz+1)+k; 
		niijpkk=i*(ny+1)*(nz+1)+(j+1)*(nz+1)+k; 
		nipjpkk=(i+1)*(ny+1)*(nz+1)+(j+1)*(nz+1)+k;
		niijjkp=i*(ny+1)*(nz+1)+j*(nz+1)+(k+1); 
		nipjjkp=(i+1)*(ny+1)*(nz+1)+j*(nz+1)+k+1; 
		niijpkp=i*(ny+1)*(nz+1)+(j+1)*(nz+1)+k+1; 
		nipjpkp=(i+1)*(ny+1)*(nz+1)+(j+1)*(nz+1)+(k+1);

		iuse=niijjkk;
		icnt=0;
		ipos=lookup_v[iuse];
		isplit=lookup_v[iuse+1]-lookup_v[iuse];
		if (isplit == 1) {
		    z = (zdata_v[ipos]+zoffset)*zscale;
		    z_sw_v[icnt] = z;
		    z_se_v[icnt] = z;
		    z_nw_v[icnt] = z;
		    z_ne_v[icnt] = z;
		}
		if (isplit == 4) {
		    z_sw_v[icnt] = (zdata_v[ipos+0]+zoffset)*zscale;
		    z_se_v[icnt] = (zdata_v[ipos+1]+zoffset)*zscale;
		    z_nw_v[icnt] = (zdata_v[ipos+2]+zoffset)*zscale;
		    z_ne_v[icnt] = (zdata_v[ipos+3]+zoffset)*zscale;
		}


		iuse=nipjjkk;
		icnt=1;
		ipos=lookup_v[iuse];
		isplit=lookup_v[iuse+1]-lookup_v[iuse];
		if (isplit == 1) {
		    z = (zdata_v[ipos]+zoffset)*zscale;
		    z_sw_v[icnt] = z;
		    z_se_v[icnt] = z;
		    z_nw_v[icnt] = z;
		    z_ne_v[icnt] = z;
		}
		if (isplit == 4) {
		    z_sw_v[icnt] = (zdata_v[ipos+0]+zoffset)*zscale;
		    z_se_v[icnt] = (zdata_v[ipos+1]+zoffset)*zscale;
		    z_nw_v[icnt] = (zdata_v[ipos+2]+zoffset)*zscale;
		    z_ne_v[icnt] = (zdata_v[ipos+3]+zoffset)*zscale;
		}


		iuse=niijpkk;
		icnt=2;
		ipos=lookup_v[iuse];
		isplit=lookup_v[iuse+1]-lookup_v[iuse];
		if (isplit == 1) {
		    z = (zdata_v[ipos]+zoffset)*zscale;
		    z_sw_v[icnt] = z;
		    z_se_v[icnt] = z;
		    z_nw_v[icnt] = z;
		    z_ne_v[icnt] = z;
		}
		if (isplit == 4) {
		    z_sw_v[icnt] = (zdata_v[ipos+0]+zoffset)*zscale;
		    z_se_v[icnt] = (zdata_v[ipos+1]+zoffset)*zscale;
		    z_nw_v[icnt] = (zdata_v[ipos+2]+zoffset)*zscale;
		    z_ne_v[icnt] = (zdata_v[ipos+3]+zoffset)*zscale;
		}


		iuse=nipjpkk;
		icnt=3;
		ipos=lookup_v[iuse];
		isplit=lookup_v[iuse+1]-lookup_v[iuse];
		if (isplit == 1) {
		    z = (zdata_v[ipos]+zoffset)*zscale;
		    z_sw_v[icnt] = z;
		    z_se_v[icnt] = z;
		    z_nw_v[icnt] = z;
		    z_ne_v[icnt] = z;
		}
		if (isplit == 4) {
		    z_sw_v[icnt] = (zdata_v[ipos+0]+zoffset)*zscale;
		    z_se_v[icnt] = (zdata_v[ipos+1]+zoffset)*zscale;
		    z_nw_v[icnt] = (zdata_v[ipos+2]+zoffset)*zscale;
		    z_ne_v[icnt] = (zdata_v[ipos+3]+zoffset)*zscale;
		}


		iuse=niijjkp;
		icnt=4;
		ipos=lookup_v[iuse];
		isplit=lookup_v[iuse+1]-lookup_v[iuse];
		if (isplit == 1) {
		    z = (zdata_v[ipos]+zoffset)*zscale;
		    z_sw_v[icnt] = z;
		    z_se_v[icnt] = z;
		    z_nw_v[icnt] = z;
		    z_ne_v[icnt] = z;
		}
		if (isplit == 4) {
		    z_sw_v[icnt] = (zdata_v[ipos+0]+zoffset)*zscale;
		    z_se_v[icnt] = (zdata_v[ipos+1]+zoffset)*zscale;
		    z_nw_v[icnt] = (zdata_v[ipos+2]+zoffset)*zscale;
		    z_ne_v[icnt] = (zdata_v[ipos+3]+zoffset)*zscale;
		}


		iuse=nipjjkp;
		icnt=5;
		ipos=lookup_v[iuse];
		isplit=lookup_v[iuse+1]-lookup_v[iuse];
		if (isplit == 1) {
		    z = (zdata_v[ipos]+zoffset)*zscale;
		    z_sw_v[icnt] = z;
		    z_se_v[icnt] = z;
		    z_nw_v[icnt] = z;
		    z_ne_v[icnt] = z;
		}
		if (isplit == 4) {
		    z_sw_v[icnt] = (zdata_v[ipos+0]+zoffset)*zscale;
		    z_se_v[icnt] = (zdata_v[ipos+1]+zoffset)*zscale;
		    z_nw_v[icnt] = (zdata_v[ipos+2]+zoffset)*zscale;
		    z_ne_v[icnt] = (zdata_v[ipos+3]+zoffset)*zscale;
		}


		iuse=niijpkp;
		icnt=6;
		ipos=lookup_v[iuse];
		isplit=lookup_v[iuse+1]-lookup_v[iuse];
		if (isplit == 1) {
		    z = (zdata_v[ipos]+zoffset)*zscale;
		    z_sw_v[icnt] = z;
		    z_se_v[icnt] = z;
		    z_nw_v[icnt] = z;
		    z_ne_v[icnt] = z;
		}
		if (isplit == 4) {
		    z_sw_v[icnt] = (zdata_v[ipos+0]+zoffset)*zscale;
		    z_se_v[icnt] = (zdata_v[ipos+1]+zoffset)*zscale;
		    z_nw_v[icnt] = (zdata_v[ipos+2]+zoffset)*zscale;
		    z_ne_v[icnt] = (zdata_v[ipos+3]+zoffset)*zscale;
		}


		iuse=nipjpkp;
		icnt=7;
		ipos=lookup_v[iuse];
		isplit=lookup_v[iuse+1]-lookup_v[iuse];
		if (isplit == 1) {
		    z = (zdata_v[ipos]+zoffset)*zscale;
		    z_sw_v[icnt] = z;
		    z_se_v[icnt] = z;
		    z_nw_v[icnt] = z;
		    z_ne_v[icnt] = z;
		}
		if (isplit == 4) {
		    z_sw_v[icnt] = (zdata_v[ipos+0]+zoffset)*zscale;
		    z_se_v[icnt] = (zdata_v[ipos+1]+zoffset)*zscale;
		    z_nw_v[icnt] = (zdata_v[ipos+2]+zoffset)*zscale;
		    z_ne_v[icnt] = (zdata_v[ipos+3]+zoffset)*zscale;
		}

    

		
		    
		zz[1]=z_ne_v[4];
		zz[2]=z_nw_v[5];
		
		zz[3]=z_se_v[6];
		zz[4]=z_sw_v[7];
		
		zz[5]=z_ne_v[0];
		zz[6]=z_nw_v[1];
		
		zz[7]=z_se_v[2];
		zz[8]=z_sw_v[3];
		
		
		
		
		if (l>=0) {
		    for (ic=1;ic<=4;ic++) {
			p_zcorn_v[ib] = zz[ic];
			ib++;
		    }
		}
		

		if (l==-1) { 
		    for (ic=5;ic<=8;ic++) {
			p_zcorn_v[ib] = zz[ic];
			ib++;
		    }
		}
				
	    }	
	}   
    }

    xtg_speak(sub,3,"--> Grid array on XTG form...DONE!");

    xtg_speak(sub,3,"Creating arrays on XTG form...DONE!");

    xtg_speak(sub,3,"Transforming to internal XTG representation ... DONE!");
    
    /* free temporary pointers */
    xtg_speak(sub,2,"Unallocating memory for tmp arrays ...");
    xtg_speak(sub,3,"Freeing lookup_v");
    free(lookup_v);
    xtg_speak(sub,3,"Freeing z_xx_v");
    free(z_nw_v);
    free(z_ne_v);
    free(z_sw_v);
    free(z_se_v);
    xtg_speak(sub,3,"Freeing *p_bot_v");
    free(xp_bot_v);
    free(yp_bot_v);
    free(zp_bot_v);
    xtg_speak(sub,3,"Freeing *p_top_v");
    free(xp_top_v);
    free(yp_top_v);
    free(zp_top_v);
    xtg_speak(sub,2,"Unallocating memory for tmp arrays ... DONE!");



    xtg_speak(sub,2,"==== _grd3d_roff_to_xtg_grid ====");

}


