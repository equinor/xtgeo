
#include <string.h>
#include <stdlib.h>
#include "libxtg.h"
#include "libxtg_.h"

/*
 ******************************************************************************
 *                      ECLIPSE GRDECL FILE
 ******************************************************************************
 *
 */


void grd3d_import_grdecl (
			  int     nx,
			  int     ny,
			  int     nz,
			  double  *p_coord_v,
			  double  *p_zcorn_v,
			  int     *p_actnum_v,
			  int     *nact,
			  char    *filename,
			  int     debug
			  )


{
    char cname[9];
    int i, j, k, kk, ib, line, nnx, nny, nnz, num_cornerlines, kzread;
    int ix, jy, kz, ier, nn;
    int nfact=0, nfzcorn=0, nfcoord=0;
    double fvalue, fvalue1, fvalue2, xmin, xmax, ymin, ymax;
    double x1, y1, x2, y2, x3, y3, cx, cy;
    int dvalue, mamode;
    FILE *fc;
    char s[24]="grd3d_import_grdecl";

    xtgverbose(debug);

    xtg_speak(s,2,"Import Eclipse GRDECL format ...");

    /* initial settings of data (used if MAPAXES)*/
    xmin=VERYLARGEFLOAT;
    ymin=VERYLARGEFLOAT;
    xmax=-1*VERYLARGEFLOAT;
    ymax=-1*VERYLARGEFLOAT;

    /* this is just a COORD sort of counter, to track if X, Y, Z is read (for xmin etc)*/
    ix=1; jy=0; kz=0;

    mamode=0;

    /*
     *-------------------------------------------------------------------------
     * Open file
     *-------------------------------------------------------------------------
     */

    xtg_speak(s,2,"Opening ASCII GRDECL file...");
    fc = x_fopen(filename, "r", debug);
    xtg_speak(s,2,"Opening file...OK!");


    /*
     *=========================================================================
     * Loop file... It is NOT necessary to do many tests; that should be done
     * by the calling PERL script?
     *=========================================================================
     */
    num_cornerlines=2*3*(nx+1)*(ny+1);
    xtg_speak(s,2,"NX NY NZ and no. of cornerlines: %d %d %d   %d",
	    nx,ny,nz,num_cornerlines);

    for (line=1;line<9999999;line++) {

	/* Get offsets */
	if (fgets(cname,9,fc) != NULL) xtg_speak(s, 4, "CNAME is:\n%s", cname);

	if (strncmp(cname,"SPECGRID",8)==0) {
	    xtg_speak(s,2,"SPECGRID found");
	    ier=fscanf(fc,"%d %d %d", &nnx, &nny, &nnz);
	    if (ier != 3) {
		xtg_error(s,"Error in reading SPECGRID");
	    }
	}

	if (strncmp(cname,"MAPAXES",7)==0) {
	    xtg_speak(s,2,"MAPAXES found");
	    ier = fscanf(fc,"%lf %lf %lf %lf %lf %lf", &x1, &y1, &x2, &y2,
                         &x3, &y3);
	    if (ier != 6) {
		xtg_error(s,"Error in reading MAPAXES");
	    }
	    mamode=1;
	}

	if (strncmp(cname,"COORD",5)==0) {
	    xtg_speak(s,2,"COORD found");
	    nfcoord=1;

	    for (i=0; i<num_cornerlines; i++) {
		if (fscanf(fc,"%lf",&fvalue) != 1)
                    xtg_error(s,"Error in reading COORD");
		if (fvalue == 9999900.0000) {
		    fvalue=-9999.99;
		}
		p_coord_v[i]=fvalue;
		if (debug >= 4) xtg_speak(s,4,"CornerLine [%d] %lf",i,fvalue);


		if (ix==1) {
		    if (p_coord_v[i]<xmin) xmin = p_coord_v[i];
		    if (p_coord_v[i]>xmax) xmax = p_coord_v[i];
		    ix=0;
		    jy=1;
		    kz=0;
		}
		else if (jy==1) {
		    if (p_coord_v[i]<ymin) ymin = p_coord_v[i];
		    if (p_coord_v[i]>ymax) ymax = p_coord_v[i];
		    ix=0;
		    jy=0;
		    kz=1;
		}
		else{
		    ix=1;
		    jy=0;
		    kz=0;
		}
	    }
	    xtg_speak(s,3,"Last value read: %lf\n",fvalue);
	}

	/*
	 * ZCORN: Eclipse has 8 corners pr cell, while XTGeo format
	 * use 4 corners (top of cell) for NZ+1 cell. This may cause
	 * problems if GAPS in GRDECL format (like BRILLD test case)
	 *
	 */

	if (strncmp(cname,"ZCORN",5)==0) {
	    xtg_speak(s,2,"ZCORN found");
	    nfzcorn=1;

	    ib=0;
	    kzread=0;
	    kk=0;
	    for (k=1; k<=2*nz; k++) {
		if (kzread==0) {
		    kzread=1;
		}
		else{
		    kzread=0;
		}
		if (k==2*nz && kzread==0) kzread=1;
		if (kzread==1) {
		    kk+=1;
		    xtg_speak(s,2,"Reading layer: %d", kk);
		}
		for (j=1; j<=ny; j++) {
		    /* "left" cell margin */
		    for (i=1; i<=nx; i++) {
			if (fscanf(fc,"%lf",&fvalue1) != 1) xtg_error(s,"Error in reading ZCORN");
			if (fscanf(fc,"%lf",&fvalue2) != 1) xtg_error(s,"Error in reading ZCORN");

			ib=x_ijk2ib(i,j,kk,nx,ny,nz+1,0);
			if (kzread==1) {
			    p_zcorn_v[4*ib+1*1-1]=fvalue1;
			    p_zcorn_v[4*ib+1*2-1]=fvalue2;
			}
		    }
		    /* "right" cell margin */
		    for (i=1; i<=nx; i++) {
			if (fscanf(fc,"%lf",&fvalue1) != 1) xtg_error(s,"Error in reading ZCORN");
			if (fscanf(fc,"%lf",&fvalue2) != 1) xtg_error(s,"Error in reading ZCORN");
			ib=x_ijk2ib(i,j,kk,nx,ny,nz+1,0);
			if (kzread==1) {
			    p_zcorn_v[4*ib+1*3-1]=fvalue1;
			    p_zcorn_v[4*ib+1*4-1]=fvalue2;
			}
		    }
		}
	    }
	    xtg_speak(s,3,"Last value read: %lf\n",fvalue2);
	}

	nn=0;
	if (strncmp(cname,"ACTNUM",6)==0) {
	    xtg_speak(s,2,"ACTNUM found");
	    nfact=1;
	    ib=0;
	    for (k=1; k<=nz; k++) {
		for (j=1; j<=ny; j++) {
		    for (i=1; i<=nx; i++) {
			if (fscanf(fc,"%d",&dvalue)==1) {
			    p_actnum_v[ib++]=dvalue;
			    if (dvalue==1) nn++;
			}
			else{
			    xtg_error(s,"Error in reading...");
			}
		    }

		}
	    }
	    xtg_speak(s,3,"Last value read: %d\n",dvalue);
	}

	if (nfact==1 && nfzcorn==1 && nfcoord==1) {
	  break;
	  /* fclose(fc); */
	}
    }
    fclose(fc);

    *nact = nn;

   /* convert from MAPAXES, if present */
    if (mamode==1) {
	xtg_speak(s,2,"Conversion via MAPAXES...");
	for (ib=0; ib<(nx+1)*(ny+1)*6; ib=ib+3) {
	    cx = p_coord_v[ib];
	    cy = p_coord_v[ib+1];
	    x_mapaxes(mamode, &cx, &cy, x1, y1, x2, y2, x3, y3, 0);
	    p_coord_v[ib]   = cx;
	    p_coord_v[ib+1] = cy;
	}
	xtg_speak(s,2,"Conversion via MAPAXES... DONE");
    }
}
