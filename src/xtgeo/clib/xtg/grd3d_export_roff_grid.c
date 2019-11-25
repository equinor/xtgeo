/*
 * ############################################################################
 * grd3d_export_roff_grid
 * Exporting a Roff grid to file
 * Current version: Keeping pillars and 4 Z corners pr layers (add 1 pseudo layer)
 * !! See also: <grd3d_export_roff_end>
 * Author: JCR
 * ############################################################################
 *
 * ############################################################################
 */


#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "libxtg.h"
#include "libxtg_.h"

/* minimum split in noded accepted to be a split-node */
#define ZMINSPLIT 0.0000

/*
 * ****************************************************************************
 *                      READ_ROFF_ASCII_GRID
 ****************************************************************************
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
 *tag subgrids
 *array int nLayers 16
 *          1            1            5            8           10           10
 *         10           15            8           20           20           20
 *          8            6            8            2
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


void grd3d_export_roff_grid (
			    int     mode,
			    int     nx,
			    int     ny,
			    int     nz,
			    int     num_subgrds,
			    int     isubgrd_to_export,
			    double   xoffset,
			    double   yoffset,
			    double   zoffset,
			    double  *p_coord_v,
			    double  *p_zcorn_v,
			    int     *p_actnum_v,
			    int     *p_subgrd_v,
			    char    *filename,
			    int     debug
			    )


{

    int    isplit, ibne, ibse, ibsw, ibnw;
    int    nzz, nzdata, ipass, myint;
    int    i, j, k, ib, ipos, nz_true;
    int    i_tmp, kftype, kc, nn;
    char   mybyte;
    int    nz1, nz2;
    double  xscale, yscale, zscale;
    float myfloat;
    double  zseb=0.0, zneb=0.0, znwb=0.0, zswb=0.0, zzzz=0.0;
    char   sub[24]="grd3d_exp_roff_grid";
    int    m;
    size_t n;

    FILE   *fc;

    xtgverbose(debug);

    xtg_speak(sub,2,"Entering routine ...");


    /*
     *-------------------------------------------------------------------------
     * Checks (mainly for debugging)
     *-------------------------------------------------------------------------
     */
    if (debug>2) {
	xtg_speak(sub,3,"Mode = %d    NX =%d    NY=%d    NZ=%d  NUM_SUBGRIDS=%d",
		  mode, nx, ny, nz, num_subgrds);

    }

    /*
     *-------------------------------------------------------------------------
     * Open file ...
     *-------------------------------------------------------------------------
     */

    xtg_speak(sub,2,"Opening ROFF file...");

    fc=fopen(filename,"wb");

    xtg_speak(sub,2,"Opening ROFF file...DONE!");

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
	    xtg_speak(sub,2,"Exporting subgrid, using K range: %d - %d",nz1,nz2);
	}
    }
    else{
	xtg_error(sub,"Fatal error: isubgrd_to_export too large");
    }


    /*
     *-------------------------------------------------------------------------
     * Header of file. Most of this is presently hardcoded
     *-------------------------------------------------------------------------
     */

    xtg_speak(sub,2,"Output ROFF header ...");


    if (mode > 0) {
	xtg_speak(sub,2,"ASCII header ...");
	m=fprintf(fc,"roff-asc\n");
	m=fprintf(fc,"#ROFF file#\n");
	m=fprintf(fc,"#Creator: GTC subsystem of GPLib by JCR#\n");
	m=fprintf(fc,"tag filedata\n");
	m=fprintf(fc,"int byteswaptest 1\n");
	m=fprintf(fc,"char filetype \"grid\"\n");
	m=fprintf(fc,"char creationDate \"01/01/2000 01:01:01\"\n");
	m=fprintf(fc,"endtag\n");
	m=fprintf(fc,"tag version\n");
	m=fprintf(fc,"int major 2\n");
	m=fprintf(fc,"int minor 0\n");
	m=fprintf(fc,"endtag\n");
	m=fprintf(fc,"tag dimensions\n");
	m=fprintf(fc,"int nX %d\n", nx);
	m=fprintf(fc,"int nY %d\n", ny);
	m=fprintf(fc,"int nZ %d\n", nz_true);
	m=fprintf(fc,"endtag\n");
	m=fprintf(fc,"tag translate\n");
	m=fprintf(fc,"float xoffset %f\n", xoffset);
	m=fprintf(fc,"float yoffset %f\n", yoffset);
	m=fprintf(fc,"float zoffset %f\n", zoffset);
	m=fprintf(fc,"endtag\n");
	m=fprintf(fc,"tag scale\n");
	xscale=1.0; yscale=1.0; zscale=-1.0;
	m=fprintf(fc,"float xscale %f\n", xscale);
	m=fprintf(fc,"float yscale %f\n", yscale);
	m=fprintf(fc,"float zscale %f\n", zscale);
	m=fprintf(fc,"endtag\n");
    }
    else{

	/* binary output */

	xtg_speak(sub,2,"BINARY header ...");

	n=fwrite("roff-bin\0",1,9,fc);

	n=fwrite("#ROFF file#\0",1,12,fc);
	n=fwrite("#Creator: CXTGeo subsystem of XTGeo by JCR#\0",1,44,fc);
	n=fwrite("tag\0filedata\0",1,13,fc);
	n=fwrite("int\0byteswaptest\0",1,17,fc); myint=1;
	n=fwrite(&myint,4,1,fc);
	n=fwrite("char\0filetype\0grid\0",1,19,fc);
	n=fwrite("char\0creationDate\0UNKNOWN\0",1,25,fc);
	n=fwrite("endtag\0",1,7,fc);
	n=fwrite("tag\0version\0",1,12,fc);
	n=fwrite("int\0major\0",1,10,fc); myint=2;
	n=fwrite(&myint,4,1,fc);
	n=fwrite("int\0minor\0",1,10,fc); myint=0;
	n=fwrite(&myint,4,1,fc);
	n=fwrite("endtag\0",1,7,fc);
	n=fwrite("tag\0dimensions\0",1,15,fc);
	n=fwrite("int\0nX\0",1,7,fc); myint=nx;
	n=fwrite(&myint,4,1,fc);
	n=fwrite("int\0nY\0",1,7,fc); myint=ny;
	n=fwrite(&myint,4,1,fc);
	n=fwrite("int\0nZ\0",1,7,fc); myint=nz_true;

	n=fwrite(&myint,4,1,fc);
	n=fwrite("endtag\0",1,7,fc);

	n=fwrite("tag\0translate\0",1,14,fc);

	n=fwrite("float\0xoffset\0",1,14,fc);
	myfloat=xoffset;
	n=fwrite(&myfloat,4,1,fc);

	n=fwrite("float\0yoffset\0",1,14,fc);
	myfloat=yoffset;
	n=fwrite(&myfloat,4,1,fc);

	n=fwrite("float\0zoffset\0",1,14,fc);
	myfloat=zoffset;
	n=fwrite(&myfloat,4,1,fc);
	n=fwrite("endtag\0",1,7,fc);

	n=fwrite("tag\0scale\0",1,10,fc);
	xscale=1.0; yscale=1.0; zscale=-1.0;
	n=fwrite("float\0xscale\0",1,13,fc);
	myfloat=xscale;
	n=fwrite(&myfloat,4,1,fc);

	n=fwrite("float\0yscale\0",1,13,fc);
	myfloat=yscale;
	n=fwrite(&myfloat,4,1,fc);

	n=fwrite("float\0zscale\0",1,13,fc);
	myfloat=zscale;
	n=fwrite(&myfloat,4,1,fc);

	n=fwrite("endtag\0",1,7,fc);
    }

    xtg_speak(sub,2,"Output ROFF header ... DONE");



    /*
     *-------------------------------------------------------------------------
     * Subgrid info
     *-------------------------------------------------------------------------
     */
    if (num_subgrds > 1 && isubgrd_to_export < 1) {
	if (mode > 0) {
	    m=fprintf(fc,"tag subgrids\n");
	    m=fprintf(fc,"array int nLayers %d\n", num_subgrds);
	}
	else{
	    n=fwrite("tag\0subgrids\0",1,13,fc);
	    n=fwrite("array\0int\0nLayers\0",1,18,fc);
	    myint=num_subgrds;
	    n=fwrite(&myint,4,1,fc);
	}
	if (mode>0){
	    i_tmp=1;
	    for (i=0;i<num_subgrds;i++) {
		m=fprintf(fc,"%6d",p_subgrd_v[i]);
	    }
	    i_tmp++;
	    if (i_tmp>6) {
		m=fprintf(fc,"\n");
		i_tmp=1;
	    }

	    if (i_tmp != 1)  m=fprintf(fc,"\n");
	    m=fprintf(fc,"endtag\n");
	}
	else{
	    for (i=0;i<num_subgrds;i++) {
		myint=p_subgrd_v[i];
		n=fwrite(&myint,4,1,fc);
	    }
	    /*n=fwrite(p_subgrd_v,4,num_subgrds,fc);*/
	    n=fwrite("endtag\0",1,7,fc);
	}
    }


    /*
     *-------------------------------------------------------------------------
     * Corner lines
     *-------------------------------------------------------------------------
     */
    if (mode>0) {
	m=fprintf(fc,"tag cornerLines\n");
	m=fprintf(fc,"array float data %d\n", (nx+1)*(ny+1)*2*3);
    }
    else{
	n=fwrite("tag\0cornerLines\0",1,16,fc);
	n=fwrite("array\0float\0data\0",1,17,fc); myint=(nx+1)*(ny+1)*2*3;
	n=fwrite(&myint,4,1,fc);
    }


    /* looping the grid and extracting bottom and top pillar XYZ */

    nzz=nz2+1;

    xtg_speak(sub,2,"CornerLines ...");




    /* printing ... and scale/translate back*/

    for (i=0; i<=nx; i++) {
	for (j=0; j<=ny; j++) {
	    ipos=6*(j*(nx+1)+i);
	    if (mode > 0) {
		m=fprintf(fc,"  %e"  ,(p_coord_v[ipos+3]/xscale)-xoffset);
		m=fprintf(fc,"  %e"  ,(p_coord_v[ipos+4]/yscale)-yoffset);
		m=fprintf(fc,"  %e"  ,(p_coord_v[ipos+5]/zscale)-zoffset);
		m=fprintf(fc,"  %e"  ,(p_coord_v[ipos+0]/xscale)-xoffset);
		m=fprintf(fc,"  %e"  ,(p_coord_v[ipos+1]/yscale)-yoffset);
		m=fprintf(fc,"  %e\n",(p_coord_v[ipos+2]/zscale)-zoffset);
	    }
	    else{
		myfloat=(p_coord_v[ipos+3]/xscale)-xoffset;
		n=fwrite(&myfloat,4,1,fc);


		myfloat=(p_coord_v[ipos+4]/yscale)-yoffset;
		n=fwrite(&myfloat,4,1,fc);
		myfloat=(p_coord_v[ipos+5]/zscale)-zoffset;
		n=fwrite(&myfloat,4,1,fc);
		myfloat=(p_coord_v[ipos+0]/xscale)-xoffset;
		n=fwrite(&myfloat,4,1,fc);
		myfloat=(p_coord_v[ipos+1]/yscale)-yoffset;
		n=fwrite(&myfloat,4,1,fc);
		myfloat=(p_coord_v[ipos+2]/zscale)-zoffset;
		n=fwrite(&myfloat,4,1,fc);

	    }
	}
    }


    if (mode>0) {
	m=fprintf(fc,"endtag\n");
    }
    else{
	n=fwrite("endtag\0",1,7,fc);
    }

    xtg_speak(sub,2,"CornerLines ...DONE!");


    xtg_speak(sub,2,"Writing splitEnz...");


    if (mode>0) {
	m=fprintf(fc,"tag zvalues\n");
	m=fprintf(fc,"array byte splitEnz %d\n",(nx+1)*(ny+1)*(nz_true+1));
	i_tmp=1;
    }
    else{
	n=fwrite("tag\0zvalues\0",1,12,fc);
	n=fwrite("array\0byte\0splitEnz\0",1,20,fc);
	myint=(nx+1)*(ny+1)*(nz_true+1);
	n=fwrite(&myint,4,1,fc);
    }


    /* Looping this twice - first splitEnz is written, next time zdata */
    for (ipass=0; ipass<=1; ipass++) {
	if (ipass==0) {
	    nzdata=0;
	}
	else{
	    if (mode>0) {
		m=fprintf(fc,"array float data %d\n",nzdata);
	    }
	    else{
		n=fwrite("array\0float\0data\0",1,17,fc);
		myint=nzdata;
		n=fwrite(&myint,4,1,fc);
	    }
	}


	for (i=0; i<=nx; i++) {
	    for (j=0; j<=ny; j++) {
		for (k=(nz2+1); k>=nz1; k--) {

		    isplit=1;

		    /* look at a node and neighbour cells */

		    /* finding splits for tops */
		    if ((i==0  && j==0) || (i==0  && j==ny) ||
			(i==nx && j==0) || (i==nx && j==ny)) {
			/* corners of grid can only av type 1 split */
			isplit=1;
			kftype=-1;
			/* need a value for later use (data array) */
			if (i==0 && j==0) {
			    ibne=x_ijk2ib(i+1,j+1,k,nx,ny,nzz,0);
			    zzzz=p_zcorn_v[4*ibne + 1*1 - 1];
			}
			if (i==0 && j==ny) {
			    ibse=x_ijk2ib(i+1,j,k,nx,ny,nzz,0);
			    zzzz=p_zcorn_v[4*ibse + 1*3 - 1];
			}
			if (i==nx && j==0) {
			    ibnw=x_ijk2ib(i,j+1,k,nx,ny,nzz,0);
			    zzzz=p_zcorn_v[4*ibnw + 1*2 - 1];
			}
			if (i==nx && j==ny) {
			    ibsw=x_ijk2ib(i,j,k,nx,ny,nzz,0);
			    zzzz=p_zcorn_v[4*ibsw + 1*4 - 1];
			}

		    }
		    else if (i==0 && (j<ny && j>0)) {
			/* go along edge of grid */
			ibse=x_ijk2ib(i+1,j,k,nx,ny,nzz,0);
			ibne=x_ijk2ib(i+1,j+1,k,nx,ny,nzz,0);

			zseb=p_zcorn_v[4*ibse + 1*3 - 1];
			zneb=p_zcorn_v[4*ibne + 1*1 - 1];
			znwb=zneb;
			zswb=zseb;
			kftype=2;

		    }
		    else if (i==nx && (j<ny && j>0)) {
			/* go along edge of grid */

			ibnw=x_ijk2ib(i,j+1,k,nx,ny,nzz,0);
			ibsw=x_ijk2ib(i,j,k,nx,ny,nzz,0);
			zswb=p_zcorn_v[4*ibsw + 1*4 - 1];
			znwb=p_zcorn_v[4*ibnw + 1*2 - 1];
			zneb=znwb;
			zseb=zswb;
			kftype=3;

		    }
		    else if (j==0 && (i<nx && i>0)) {
			/* go along edge of grid */

			ibnw=x_ijk2ib(i,j+1,k,nx,ny,nzz,0);
			ibne=x_ijk2ib(i+1,j+1,k,nx,ny,nzz,0);
			znwb=p_zcorn_v[4*ibnw + 1*2 - 1];
			zneb=p_zcorn_v[4*ibne + 1*1 - 1];
			zseb=zneb;
			zswb=znwb;
			kftype=4;

		    }
		    else if (j==ny && (i<nx && i>0)) {
			/* go along edge of grid */
			ibsw=x_ijk2ib(i,j,k,nx,ny,nzz,0);
			ibse=x_ijk2ib(i+1,j,k,nx,ny,nzz,0);
			zswb=p_zcorn_v[4*ibsw + 1*4 - 1];
			zseb=p_zcorn_v[4*ibse + 1*3 - 1];
			zneb=zseb;
			znwb=zswb;
			kftype=5;

		    }
		    else{
			ibnw=x_ijk2ib(i,j+1,k,nx,ny,nzz,0);
			ibsw=x_ijk2ib(i,j,k,nx,ny,nzz,0);
			ibse=x_ijk2ib(i+1,j,k,nx,ny,nzz,0);
			ibne=x_ijk2ib(i+1,j+1,k,nx,ny,nzz,0);

			/* southwest of node i,j is ne (crn 4) of cell i-1,j-1 */
			/* below, ETC */

			/* look at bottom part of cell: */
			zswb=p_zcorn_v[4*ibsw + 1*4 - 1];
			zseb=p_zcorn_v[4*ibse + 1*3 - 1];
			znwb=p_zcorn_v[4*ibnw + 1*2 - 1];
			zneb=p_zcorn_v[4*ibne + 1*1 - 1];


			kftype=6;
		    }

		    if (kftype > 0) {
			if ((fabs(zswb-zseb)>ZMINSPLIT) ||
			    (fabs(zseb-zneb)>ZMINSPLIT) ||
			    (fabs(znwb-zneb)>ZMINSPLIT) ||
			    (fabs(zswb-znwb)>ZMINSPLIT) ||
			    (fabs(zseb-znwb)>ZMINSPLIT) ||
			    (fabs(zswb-zneb)>ZMINSPLIT)){

			    isplit=4;


			}
			else{
			    zzzz=zswb; /* just choose one */
			}
		    }

		    if (isplit==4 && debug >= 4) {
			xtg_speak(sub,4,"-------------------------------");
			xtg_speak(sub,4,"SPLIT type 4:");
			xtg_speak(sub,4,"Node ijk: %d %d %d", i, j, k);
			xtg_speak(sub,4,"nw ne se sw: %f %f %f %f",
				znwb, zneb, zseb, zswb);
			xtg_speak(sub,4,"Split sub-type: %d", kftype);
		    }


		    /* printing! */


		    if (ipass == 0) {
			if (mode>0) {
			    m=fprintf(fc,"%4d",isplit);
			    i_tmp++;
			    if (i_tmp > 12) {
				i_tmp=1;
				m=fprintf(fc,"\n");
			    }
			}
			else{
			    mybyte=isplit;
			    n=fwrite(&mybyte,1,1,fc);
			}
			/* collecting the number of zdata */
			nzdata+=isplit;
		    }
		    else{
			/* collect and print zdata */
			if (isplit==4) {
			    if (mode>0) {
				m=fprintf(fc,"  %e"  ,(zswb/zscale)-zoffset);
				m=fprintf(fc,"  %e"  ,(zseb/zscale)-zoffset);
				m=fprintf(fc,"  %e"  ,(znwb/zscale)-zoffset);
				m=fprintf(fc,"  %e\n",(zneb/zscale)-zoffset);

			    }
			    else{
				myfloat=(zswb/zscale)-zoffset;
				n=fwrite(&myfloat,4,1,fc);
				myfloat=(zseb/zscale)-zoffset;
				n=fwrite(&myfloat,4,1,fc);
				myfloat=(znwb/zscale)-zoffset;
				n=fwrite(&myfloat,4,1,fc);
				myfloat=(zneb/zscale)-zoffset;
				n=fwrite(&myfloat,4,1,fc);
			    }
			}
			else{
			    if (mode>0) {
				m=fprintf(fc,"  %e\n",(zzzz/zscale)-zoffset);
			    }
			    else{
				myfloat=(zzzz/zscale)-zoffset;
				n=fwrite(&myfloat,4,1,fc);
			    }
			}
		    }
		}
	    }
	}
	if (mode>0 && i_tmp != 1) m=fprintf(fc,"\n");
	i_tmp=1;
    }
    if (mode>0) {
	m=fprintf(fc,"endtag\n");
    }
    else{
	n=fwrite("endtag\0",1,7,fc);
    }



    xtg_speak(sub,2,"Writing zdata...DONE!");

    xtg_speak(sub,2,"Writing <active>...!");
    xtg_speak(sub,3,"NZ is %d",nz_true);
    nn=nx*ny*nz_true;

    if (mode>0) {
	m=fprintf(fc,"tag active\n");
	m=fprintf(fc,"array bool data %d\n",nn);
    }
    else{
	n=fwrite("tag\0active\0",1,11,fc);
	n=fwrite("array\0bool\0data\0",1,16,fc);
	myint=nn;
	n=fwrite(&myint,4,1,fc);
    }
    i_tmp=1;
    for (i=1;i<=nx;i++) {
	for (j=1;j<=ny;j++) {
	    for (k=nz2;k>=nz1;k--) {
		ib=x_ijk2ib(i,j,k,nx,ny,nz,0);

		if (mode>0) {
		    m=fprintf(fc,"%2d",p_actnum_v[ib]);
		    i_tmp++;
		    if (i_tmp > 12) {
			i_tmp=1;
			m=fprintf(fc,"\n");
		    }
		}
		else{
		    mybyte=p_actnum_v[ib];
		    n=fwrite(&mybyte,1,1,fc);
		}
	    }
	}
    }

    if (mode>0 && i_tmp!=1)m=fprintf(fc,"\n");

    if (mode>0) {
	m=fprintf(fc,"endtag\n");
    }
    else{
	n=fwrite("endtag\0",1,7,fc);
    }
    xtg_speak(sub,2,"Writing <active>...DONE!");


    fclose(fc);


    xtg_speak(sub,2,"==== Exiting <grd3d_export_roff_grid> ====");

}
