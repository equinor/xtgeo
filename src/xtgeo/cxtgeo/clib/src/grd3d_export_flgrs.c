/*
 * ############################################################################
 * grd3d_export_flgrs.c
 * Gives both cell along a fault an LGR to that cab be nonuniform
 * All other cells are 0
 * Input:
 *          nx
 *          ny
 *          nz
 *          p_coord_v
 *          pzcorn_v
 *          p_actnum_v
 *          flimit      = minimum throw to be regarded as throw
 *          file        = name of file to write to
 *          append      = flag = 1 if append to given file name (a GRDECL file)
 *
 * Output:
 *          Will export CARFIN keys on GRDECL formats
 *
 * Returns:
 *          Nothing
 *
 * Author:  JCR
 * ############################################################################
 */

#include <math.h>
#include "libxtg.h"
#include "libxtg_.h"

/* protos for local functions */
void carfin16 (FILE *fc, int lgr, int *ibstart, int *idim, int *jdim, int *kdim, int *ftype,
	       int *actnum, int nx, int ny, int nz, int mdiv);


void grd3d_export_flgrs(
			int nx,
			int ny,
			int nz,
			double *p_coord_v,
			double *p_zcorn_v,
			int   *p_actnum_v,
			double flimit,
			char  *file,
			int   mdiv,
			int   kthrough,
			int   append,
			int   additional,
			int   debug
		       )

{
    /* locals */

    int   i, j, k, ia, ib, ip, iq, ix, k1, k2, k3, k4, itype, ityp1, ityp2, ifound;
    int   ii, jj, kk, ip1, ip2, jlgr1,jlgr2, ja1, ja2, lgrcount, lgr, lgr2;
    int   b1, b2, b3, b4;
    int   *p_fau_v, *p_lgr_v, *p_lgrstart_v, *p_lgridim_v, *p_lgrjdim_v, *p_lgrkdim_v;
    int   *k1_v=NULL, *k2_v=NULL, *k3_v=NULL, *k4_v=NULL;
    int   *b1_v=NULL, *b2_v=NULL, *b3_v=NULL, *b4_v=NULL;
    double *p_use_v;


    char  s[24]="grd3d_export_flgrs";
    FILE  *fc;

    xtgverbose(debug);
    xtg_speak(s,2,"Entering <grd3d_export_flgrs>");
    xtg_speak(s,3,"NX NY NZ: %d %d %d", nx, ny, nz);
    xtg_speak(s,2,"Fault limit throw is %f", flimit);

    /* flimit=0.5; */


    p_fau_v=calloc(nx*ny*nz,sizeof(int));
    p_lgr_v=calloc(nx*ny*nz,sizeof(int));
    p_use_v=calloc(nx*ny*nz,sizeof(double));


    /* corner type 0 = no split, 1=I, 2=J, 3=both, 4=diagonal split */

    if (additional==1) {
	k1_v=calloc(nx*ny*nz,sizeof(int));
	k2_v=calloc(nx*ny*nz,sizeof(int));
	k3_v=calloc(nx*ny*nz,sizeof(int));
	k4_v=calloc(nx*ny*nz,sizeof(int));

	/* border split, 1=on; B1=lower, B2=upper, B3=left, B4=right */
	b1_v=calloc(nx*ny*nz,sizeof(int));
	b2_v=calloc(nx*ny*nz,sizeof(int));
	b3_v=calloc(nx*ny*nz,sizeof(int));
	b4_v=calloc(nx*ny*nz,sizeof(int));
    }

    /*initiliasing to 0 is done in Perl ptrcreate*/

    xtg_speak(s,1,"Looping cells and finding faults...");
    ia=1;
    for (k = 1; k <= nz; k++) {
	xtg_speak(s,2,"Finished layer %d of %d",k,nz);
	/* loop in X */
	for (j = 1; j <= ny; j++) {
	    for (i = 1; i <= nx; i++) {
		/* parameter counting */
		ip=x_ijk2ib(i,j,k,nx,ny,nz,0);
		if (p_actnum_v[ip]==1) {

		    /* the normal case is a cell surrounded by other cells */
		    /* all other are exceptions.... */
		    k1=-1; k2=-1; k3=-1; k4=-1;

		    /* neighbour i+1 */
		    /* 2 vs 1 */
		    if (i<nx){
		        iq=x_ijk2ib(i+1,j,k,nx,ny,nz,0);
			if (p_actnum_v[iq]==1){ ia=1; } else {ia=0;};
			if (fabs(p_zcorn_v[4*ip+2-1]-p_zcorn_v[4*iq+1-1])>flimit) {
			    if (ia==1) {
				k2=1;
			    }
			}
			/* i+1 4 vs 3 */
			if (fabs(p_zcorn_v[4*ip+4-1]-p_zcorn_v[4*iq+3-1])>flimit){
			    if (ia==1) {
				k4=1;
			    }
			}
		    }


		    /* neighbour i-1 */
		    /* 1 vs 2 */
		    if (i>1){
		        iq=x_ijk2ib(i-1,j,k,nx,ny,nz,0);
			if (p_actnum_v[iq]==1){ ia=1; } else {ia=0;};
			if (fabs(p_zcorn_v[4*ip+1-1]-p_zcorn_v[4*iq+2-1])>flimit) {
			    if (ia==1) {
				k1=1;
			    }
			}
			/* 3 vs 4 */
			if (fabs(p_zcorn_v[4*ip+3-1]-p_zcorn_v[4*iq+4-1])>flimit){
			    if (ia==1) {
				k3=1;
			    }
			}
		    }


		    /* neighbour j-1 */
		    /* 1 vs 3 */
		    if (j>1){
			iq=x_ijk2ib(i,j-1,k,nx,ny,nz,0);
			if (p_actnum_v[iq]==1){ ia=1; } else {ia=0;};
			if (fabs(p_zcorn_v[4*ip+1-1]-p_zcorn_v[4*iq+3-1])>flimit) {
			    if (ia==1 && k1>=0) {
				k1=k1+2;
			    }
			    else if (ia==1 && k1<0){
				k1=2;
			    }
			    else if (ia==0 && k1>0){
				k1=1;
			    }
			    else{
				k1=-2;
			    }
			}
			/* 2 vs 4 */
			if (fabs(p_zcorn_v[4*ip+2-1]-p_zcorn_v[4*iq+4-1])>flimit){
			    if (ia==1 && k2>=0) {
				k2=k2+2;
			    }
			    else if (ia==1 && k2<0){
				k2=2;
			    }
			    else if (ia==0 && k2>0){
				k2=1;
			    }
			    else{
				k2=-2;
			    }
			}
		    }


		    /* neighbour j+1 */
		    /* 3 vs 1 */
		    if (j<ny) {
			iq=x_ijk2ib(i,j+1,k,nx,ny,nz,0);
			if (p_actnum_v[iq]==1){ ia=1; } else {ia=0;};
			if (fabs(p_zcorn_v[4*ip+3-1]-p_zcorn_v[4*iq+1-1])>flimit) {
			    if (ia==1 && k3>=0) {
				k3=k3+2;
			    }
			    else if (ia==1 && k3<0){
				k3=2;
			    }
			    else if (ia==0 && k3>0){
				k3=1;
			    }
			    else{
				k3=-2;
			    }
			}
			/* 4 vs 2 */
			if (fabs(p_zcorn_v[4*ip+4-1]-p_zcorn_v[4*iq+2-1])>flimit){
			    if (ia==1 && k4>=0) {
				k4=k4+2;
			    }
			    else if (ia==1 && k4<0){
				k4=2;
			    }
			    else if (ia==0 && k4>0){
				k4=1;
			    }
			    else{
				k4=-2;
			    }
			}
		    }


		    /* Next are the corners --------------------------------------- */

		    /* neighbour i-1,j-1 */
		    /* 1 vs 4 */
		    ix=0;
		    if (i>1 && j>1) {
			iq=x_ijk2ib(i-1,j-1,k,nx,ny,nz,0);
			if (p_actnum_v[iq]==1){ ia=1; } else {ia=0;};
			if (ia==1 && fabs(p_zcorn_v[4*ip+1-1]-p_zcorn_v[4*iq+4-1])>flimit) {
			    ix=1;
			}
			if (ix==1 && k1<0) {
			    k1=4;
			}
		    }

		    /* neighbour i+1,j-1 */
		    /* 2 vs 3 */
		    ix=0;
		    if (i<nx && j>1) {
			iq=x_ijk2ib(i+1,j-1,k,nx,ny,nz,0);
			if (p_actnum_v[iq]==1){ ia=1; } else {ia=0;};
			if (ia==1 && fabs(p_zcorn_v[4*ip+2-1]-p_zcorn_v[4*iq+3-1])>flimit) {
			    ix=1;
			}

			if (ix==1 && k2<0) {
			    k2=4;
			}
		    }

		    /* neighbour i-1,j+1 */
		    /* 3 vs 2 */
		    ix=0;
		    if (i>1 && j<ny) {
			iq=x_ijk2ib(i-1,j+1,k,nx,ny,nz,0);
			if (p_actnum_v[iq]==1){ ia=1; } else {ia=0;};
			if (ia==1 && fabs(p_zcorn_v[4*ip+3-1]-p_zcorn_v[4*iq+2-1])>flimit) {
			    ix=1;
			}
			if (ix==1 && k3<0) {
			    k3=4;
			}

		    }
		    /* neighbour i+1,j+1 */
		    /* 4 vs 1 */
		    ix=0;
		    if (i<nx && j<ny) {
			iq=x_ijk2ib(i+1,j+1,k,nx,ny,nz,0);
			if (p_actnum_v[iq]==1){ ia=1; } else {ia=0;};
			if (ia==1 && fabs(p_zcorn_v[4*ip+4-1]-p_zcorn_v[4*iq+1-1])>flimit) {
			    ix=1;
			}
			if (ix==1 && k4<0) {
			    k4=4;
			}
		    }

		    if (k1<0) k1=0;
		    if (k2<0) k2=0;
		    if (k3<0) k3=0;
		    if (k4<0) k4=0;


		    /*
		     * ------------------------------------------------------------------------------
		     * Now make a classification of that cell border, an integer.
		     * ------------------------------------------------------------------------------
		     */
		    b1=0; b2=0; b3=0; b4=0;

		    if (k1==2 || k1==3) b1=1;
		    if (k2==3 || k2==2) b1=1;
		    if (k3==2 || k3==3) b2=1;
		    if (k4==2 || k4==3) b2=1;
		    if (k1==1 || k1==3) b3=1;
		    if (k3==1 || k3==3) b3=1;
		    if (k2==1 || k2==3) b4=1;
		    if (k4==1 || k4==3) b4=1;

		    /*
		     * ------------------------------------------------------------------------------
		     * Now make a classification of that cell, an integer. There are really many!
		     * ------------------------------------------------------------------------------
		     */

		    xtg_speak(s,2,"Classifying for I J K = %d %d %d ...",i,j,k);

		    itype=0;
		    /* one face */
		    if (b1==0 && b2==0 && b3==1 && b4==0) itype=1;
		    if (b1==0 && b2==0 && b3==0 && b4==1) itype=2;
		    if (b1==1 && b2==0 && b3==0 && b4==0) itype=3;
		    if (b1==0 && b2==1 && b3==0 && b4==0) itype=4;

		    /* one corner; adjacent faces */
		    if (b1==0 && b2==1 && b3==1 && b4==0) itype=5;
		    if (b1==0 && b2==1 && b3==0 && b4==1) itype=6;
		    if (b1==1 && b2==0 && b3==1 && b4==0) itype=7;
		    if (b1==1 && b2==0 && b3==0 && b4==1) itype=8;


		    if ((k1==0 && k2==0 && k3==4 && k4==0))itype=9;
		    if ((k1==0 && k2==0 && k3==0 && k4==4))itype=10;
		    if ((k1==4 && k2==0 && k3==0 && k4==0))itype=11;
		    if ((k1==0 && k2==4 && k3==0 && k4==0))itype=12;


		    if (b1==1 && b2==1 && b3==1 && b4==0) itype=13;
		    if (b1==0 && b2==1 && b3==1 && b4==1) itype=14;
		    if (b1==1 && b2==1 && b3==0 && b4==1) itype=15;
		    if (b1==1 && b2==0 && b3==1 && b4==1) itype=16;

		    if (b1==0 && b2==0 && b3==1 && b4==1) itype=17;
		    if (b1==1 && b2==1 && b3==0 && b4==0) itype=18;

		    if (itype==5 && k2==4) itype=19;
		    if (itype==6 && k1==4) itype=20;
		    if (itype==8 && k3==4) itype=21;
		    if (itype==7 && k4==4) itype=22;

		    if (b1==1 && b2==1 && b3==1 && b4==1) itype=23;

		    if (itype==1 && k2==4 && k4!=4) itype=24;
		    if (itype==4 && k1==4 && k2!=4) itype=25;
		    if (itype==2 && k3==4 && k1!=5) itype=26;
		    if (itype==3 && k4==4 && k3!=4) itype=27;

		    if (itype==1 && k2!=4 && k4==4) itype=28;
		    if (itype==4 && k1!=4 && k2==4) itype=29;
		    if (itype==2 && k3!=4 && k1==4) itype=30;
		    if (itype==3 && k4!=4 && k3==4) itype=31;

		    if (itype==1 && k2==4 && k4==4) itype=32;
		    if (itype==4 && k1==4 && k2==4) itype=33;
		    if (itype==2 && k3==4 && k1==4) itype=34;
		    if (itype==3 && k4==4 && k3==4) itype=35;

		    if (k3==4 && k4==4 && k1==0 && k2==0) itype=36;
		    if (k3==0 && k4==4 && k1==0 && k2==4) itype=37;
		    if (k3==0 && k4==0 && k1==4 && k2==4) itype=38;
		    if (k3==4 && k4==0 && k1==4 && k2==0) itype=39;

		    if (k3==4 && k4==4 && k1==0 && k2==4) itype=40;
		    if (k3==0 && k4==4 && k1==4 && k2==4) itype=41;
		    if (k3==4 && k4==4 && k1==4 && k2==0) itype=42;
		    if (k3==4 && k4==0 && k1==4 && k2==4) itype=43;

		    if (k3==4 && k4==4 && k1==4 && k2==4) itype=44;

		    if (k3==4 && k4==0 && k1==0 && k2==4) itype=45;
		    if (k3==0 && k4==4 && k1==4 && k2==0) itype=46;


		    p_fau_v[ip]=itype;

		    if (additional==1) {
			k1_v[ip]=k1;
			k2_v[ip]=k2;
			k3_v[ip]=k3;
			k4_v[ip]=k4;

			b1_v[ip]=b1;
			b2_v[ip]=b2;
			b3_v[ip]=b3;
			b4_v[ip]=b4;
		    }

		    xtg_speak(s,2,"Classifying for I J K = %d %d %d done, ITYPE = %d",i,j,k,itype);
		    xtg_speak(s,2,"IP and IA %d  %d",ip,ia);


		}
	    }
	}
    }

    /*
     * Now it is possible we want LGR's to be from 1 to NZ anyway. So I loop to check
     * cells, and ftype them if necessary. This is optional.
     * The type of fault is special: type=-1, .... What about UNDEF cells??
     */

    if (kthrough==1) {
	xtg_speak(s,1,"Include all cells in a vertical stack...");
	ia=1;
	for (j = 1; j <= ny; j++) {
	    for (i = 1; i <= nx; i++) {
		for (k = 1; k <= nz; k++) {
		    /* parameter counting */
		    ip=x_ijk2ib(i,j,k,nx,ny,nz,0);
		    if (p_actnum_v[ip]==1 && p_fau_v[ip] > 0) {
			for (kk = 1; kk <= nz; kk++) {
			    ib=x_ijk2ib(i,j,kk,nx,ny,nz,0);
			    if (p_actnum_v[ib]==1 && p_fau_v[ib] == 0) {
				p_fau_v[ib]=49;
			    }
			}
		    }
		}
	    }
	}
    }


    /* write a global parameter called FTYPE (itype) */

    x_conv_int2double(nx*ny*nz,p_fau_v,p_use_v,debug);

    grd3d_export_grdeclprop(nx,ny,nz,1,"FTYPE",p_use_v,file,append,debug);

    if (additional==1) {

	x_conv_int2double(nx*ny*nz,k1_v,p_use_v,debug);
	grd3d_export_grdeclprop(nx,ny,nz,1,"K1",p_use_v,file,append,debug);

	x_conv_int2double(nx*ny*nz,k2_v,p_use_v,debug);
	grd3d_export_grdeclprop(nx,ny,nz,1,"K2",p_use_v,file,append,debug);

	x_conv_int2double(nx*ny*nz,k3_v,p_use_v,debug);
	grd3d_export_grdeclprop(nx,ny,nz,1,"K3",p_use_v,file,append,debug);

	x_conv_int2double(nx*ny*nz,k4_v,p_use_v,debug);
	grd3d_export_grdeclprop(nx,ny,nz,1,"K4",p_use_v,file,append,debug);

	x_conv_int2double(nx*ny*nz,b1_v,p_use_v,debug);
	grd3d_export_grdeclprop(nx,ny,nz,1,"B1",p_use_v,file,append,debug);

	x_conv_int2double(nx*ny*nz,b2_v,p_use_v,debug);
	grd3d_export_grdeclprop(nx,ny,nz,1,"B2",p_use_v,file,append,debug);

	x_conv_int2double(nx*ny*nz,b3_v,p_use_v,debug);
	grd3d_export_grdeclprop(nx,ny,nz,1,"B3",p_use_v,file,append,debug);

	x_conv_int2double(nx*ny*nz,b4_v,p_use_v,debug);
	grd3d_export_grdeclprop(nx,ny,nz,1,"B4",p_use_v,file,append,debug);

    }

    /*
     *===========================================================================
     * Group LGR's
     * Identinty ijk corners that groups LGR's as 2 in I width and at least
     * 1 in J dir
     *===========================================================================
     */

    for (ip = 0; ip <= nx*ny*nz; ip++) p_lgr_v[ip]=0;

    xtg_speak(s,1,"Grouping LGRs ...");

    lgrcount=0;
    for (k = 1; k <= nz; k++) {
	for (j = 1; j <= ny; j++) {
	    for (i = 1; i <= nx; i++) {
		/* look to neighbour */
		if (i < nx) {
		    /* walk in K and then J direction to find I pairs */
		    ifound=0;
		    for (jj = j; jj <= ny; jj++) {
		        for (kk = k; kk <= nz; kk++) {
			    ii=i;
			    ip1=x_ijk2ib(ii,jj,kk,nx,ny,nz,0);
			    ja1=p_actnum_v[ip1];
			    jlgr1=p_lgr_v[ip1];
			    ityp1=p_fau_v[ip1];

			    ii=i+1;
			    ip2=x_ijk2ib(ii,jj,kk,nx,ny,nz,0);
			    ja2=p_actnum_v[ip2];
			    jlgr2=p_lgr_v[ip2];
			    ityp2=p_fau_v[ip2];


			    if ((jlgr1==0 && jlgr2==0) && (ityp1>0 && ityp2>0)) {
				if (ifound==0) {
				    lgrcount++;
				    ifound=2;
				}
				else if (ifound==1) {
				    ifound=2;
				}
				p_lgr_v[ip1]=lgrcount;
				p_lgr_v[ip2]=lgrcount;

			    }
			    else{
				if (ifound>0 && ja1==0 && ja2==0) {
				    ifound=1;
				}
				else{
				    ifound=0;
				}
			    }
			}
			if (ifound>0) ifound=1;  /* means that next j pair is potentially included */
		    }
		}
	    }
	}
    }

    xtg_speak(s,1,"Groups of pair-wise I LGRs: %d", lgrcount-1);
    if (additional==1) {
	x_conv_int2double(nx*ny*nz,p_lgr_v,p_use_v,debug);
	grd3d_export_grdeclprop(nx,ny,nz,1,"LGR1",p_use_v,file,append,debug);
    }


    /* now find lonely cells rows in I dir */

    for (k = 1; k <= nz; k++) {
	for (j = 1; j <= ny; j++) {
	    for (i = 1; i <= nx; i++) {
		/* look to neighbour */
		ifound=0;
		for (jj = j; jj <= ny; jj++) {
		    for (kk = k; kk <= nz; kk++) {
			ii=i;
			ip1=x_ijk2ib(ii,jj,kk,nx,ny,nz,0);
			ja1=p_actnum_v[ip1];
			jlgr1=p_lgr_v[ip1];
			ityp1=p_fau_v[ip1];

			if ((jlgr1==0) && (ityp1>0)) {
			    if (ifound==0) {
				lgrcount++;
				ifound=2;
			    }
			    else if (ifound==1) {
				ifound=2;
			    }
			    p_lgr_v[ip1]=lgrcount;

			}
			else{
			    if (ifound>0 && ja1==0) {
				ifound=1;
			    }
			    else{
				ifound=0;
			    }
			}
		    }
		    if (ifound>0) ifound=1;  /* means that next j is potentially included */
		}
	    }
	}
    }




    x_conv_int2double(nx*ny*nz,p_lgr_v,p_use_v,debug);
    grd3d_export_grdeclprop(nx,ny,nz,1,"LGR_NUMBER",p_use_v,file,append,debug);


    xtg_speak(s,1,"Grouping LGRs ... DONE ... %d LGR groups", lgrcount);

    /*
     *===========================================================================
     * Loop thorugh LGR groups and find start point and dimensions
     *===========================================================================
     */

    xtg_speak(s,1,"Allocating memory....");

    p_lgrstart_v=calloc(lgrcount+1,sizeof(int));
    p_lgridim_v=calloc(lgrcount+1,sizeof(int));
    p_lgrjdim_v=calloc(lgrcount+1,sizeof(int));
    p_lgrkdim_v=calloc(lgrcount+1,sizeof(int));


    xtg_speak(s,1,"Finding dimensions for LGR groups...");

    for (ii = 0; ii <= lgrcount; ii++) p_lgrstart_v[ii]=-1;  /* will be ib cell count */
    for (ii = 0; ii <= lgrcount; ii++) p_lgridim_v[ii]=1;    /* will be local I count */
    for (ii = 0; ii <= lgrcount; ii++) p_lgrjdim_v[ii]=1;    /* etc */
    for (ii = 0; ii <= lgrcount; ii++) p_lgrkdim_v[ii]=1;

    for (k = 1; k <= nz; k++) {
	for (j = 1; j <= ny; j++) {
	    for (i = 1; i <= nx; i++) {
		ib=x_ijk2ib(i,j,k,nx,ny,nz,0);
		lgr=p_lgr_v[ib];
		if (lgr > 0 && p_lgrstart_v[lgr] < 0) {
		    p_lgrstart_v[lgr]=ib;

		    /* all LGRs are rectangular */
		    for (kk = k; kk <= nz; kk++) {
			ip=x_ijk2ib(i,j,kk,nx,ny,nz,0);
			lgr2=p_lgr_v[ip];
			if (lgr2==lgr) {
			    p_lgrkdim_v[lgr]=kk-k+1;
			}
		    }

		    for (jj = j; jj <= ny; jj++) {
			ip=x_ijk2ib(i,jj,k,nx,ny,nz,0);
			lgr2=p_lgr_v[ip];
			if (lgr2==lgr) {
			    p_lgrjdim_v[lgr]=jj-j+1;
			}
		    }

		    for (ii = i; ii <= nx; ii++) {
			ip=x_ijk2ib(ii,j,k,nx,ny,nz,0);
			lgr2=p_lgr_v[ip];
			if (lgr2==lgr) {
			    p_lgridim_v[lgr]=ii-i+1;
			}
		    }

		}
	    }
	}
    }



    /*
     *===========================================================================
     * Open file; prepare for write, and write LGRs
     *===========================================================================
     */

    xtg_speak(s,1,"Write LGRs ...");

    if (append==1) {
	fc=fopen(file,"ab");
    }
    else{
	fc=fopen(file,"wb");
    }

    for (lgr=1; lgr <= lgrcount; lgr++) {
	carfin16(fc,lgr,p_lgrstart_v,p_lgridim_v,p_lgrjdim_v,p_lgrkdim_v, p_fau_v,
		 p_actnum_v, nx, ny, nz, mdiv);
    }

    xtg_speak(s,1,"Write LGRs ... DONE!");

    xtg_speak(s,1,"Data written to Eclipse GRDECL file: %s", file);

    fclose(fc);


    /*
     *===========================================================================
     * Free temporary pointers ... coming
     *===========================================================================
     */

}



void carfin16 (FILE *fc, int lgr, int *ibstart, int *idim, int *jdim, int *kdim, int *ftype,
	       int *actnum, int nx, int ny, int nz, int mdiv)
{
    char lgrname[8], keywx[8], keywy[8];
    int  m1, m2, m3, m4, im, jm, km, ii, jj, kk, ib, iq, ix, jy, kz, iii, jjj, kkk;
    int  i, j, k, l, mcount, itype, imm, jmm, kmm, iv, jv;

    ix = 0;
    jy = 0;
    kz = 0;

    m1=1;
    m2=(mdiv/2) - 1;
    m3=m2;
    m4=1;
    im=4;
    jm=4;
    km=1;

    strcpy(keywx,"HXFIN");
    strcpy(keywy,"HYFIN");


    /* find cell to start from, and relative dimensions */

    ib=ibstart[lgr];
    x_ib2ijk(ib,&i,&j,&k,nx,ny,nz,0);
    ii=idim[lgr];
    jj=jdim[lgr];
    kk=kdim[lgr];

    imm=im*ii;
    jmm=jm*jj;
    kmm=km*kk;



    sprintf(lgrname,"F%06d",lgr);
    fprintf(fc,"\nCARFIN\n %s %6d %6d %6d %6d %6d %6d %6d %6d %6d / \n",
	    lgrname,i,i+ii-1,j,j+jj-1,k,k+kk-1,imm,jmm,kmm);
    fprintf(fc,"%s\n",keywx);
    for (l=1;l<=ii;l++) {
	fprintf(fc,"%6d %6d %6d %6d \n",m1,m2,m3,m4);
    }
    fprintf(fc,"/\n");

    fprintf(fc,"%s\n",keywy);
    for (l=1;l<=jj;l++) {
	fprintf(fc,"%6d %6d %6d %6d \n",m1,m2,m3,m4);
    }
    fprintf(fc,"/\n");



    mcount=0;
    fprintf(fc,"\nFTYPE\n");

    ib=x_ijk2ib(ix,jy,kz,nx,ny,nz,0);
    for (kkk=1; kkk<=kmm; kkk++) {
	jv=0;
	for (jjj=1; jjj<=jmm; jjj++) {
	    jv++;
	    if (jv>4) jv=1;
	    iv=0;
	    for (iii=1; iii<=imm; iii++) {
		iv++;
		if (iv>4) iv=1;

		iq=0;

		/*
		 * each global cell is divided in 4 in I and J,
		 * need to find the corresponding global cell
		 * to derive which ftype
		 */

		ix=i+((iii-1)/4);
		jy=j+((jjj-1)/4);
		kz=k+kkk-1;

		ib=x_ijk2ib(ix,jy,kz,nx,ny,nz,0);


		itype=ftype[ib];

		if (itype==1 && iv==1) iq=51;

		if (itype==2 && iv==4) iq=51;
		if (itype==3 && jv==1) iq=52;
		if (itype==4 && jv==4) iq=52;

		if (itype==5 && iv==1) iq=51;
		if (itype==5 && jv==4) iq=52;
		if (itype==5 && iv==1 && jv==4) iq=53;

		if (itype==6 && iv==4) iq=51;
		if (itype==6 && jv==4) iq=52;
		if (itype==6 && iv==4 && jv==4) iq=53;


		if (itype==7 && iv==1) iq=51;
		if (itype==7 && jv==1) iq=52;
		if (itype==7 && iv==1 && jv==1) iq=53;

		if (itype==8 && iv==4) iq=51;
		if (itype==8 && jv==1) iq=52;
		if (itype==8 && iv==4 && jv==1) iq=53;


		if (itype==9  && iv==1 && jv==4) iq=53;
		if (itype==10 && iv==4 && jv==4) iq=53;
		if (itype==11 && iv==1 && jv==1) iq=53;
		if (itype==12 && iv==4 && jv==1) iq=53;

		if (itype==13 && iv==1) iq=51;
		if (itype==13 && (jv==1 || jv==4)) iq=52;
		if (itype==13 && iv==1 && (jv==1 || jv==4)) iq=53;

		if (itype==14 && (iv==1 || iv==4)) iq=51;
		if (itype==14 && jv==4) iq=52;
		if (itype==14 && (iv==1 || iv==4) && jv==4) iq=53;

		if (itype==15 && iv==4) iq=51;
		if (itype==15 && (jv==1 || jv==4)) iq=52;
		if (itype==15 && iv==4 && (jv==1 || jv==4)) iq=53;

		if (itype==16 && (iv==1 || iv==4)) iq=51;
		if (itype==16 && jv==1) iq=52;
		if (itype==16 && iv==4 && (jv==1 || jv==4)) iq=53;

		if (itype==17 && (iv==1 || iv==4)) iq=51;
		if (itype==18 && (jv==1 || jv==4)) iq=52;

		if (itype==19 && iv==1) iq=51;
		if (itype==19 && jv==4) iq=52;
		if (itype==19 && iv==1 && jv==4) iq=53;
		if (itype==19 && iv==4 && jv==1) iq=53;

		if (itype==20 && iv==4) iq=51;
		if (itype==20 && jv==4) iq=52;
		if (itype==20 && iv==4 && jv==4) iq=53;
		if (itype==20 && iv==1 && jv==1) iq=53;

		if (itype==21 && iv==4) iq=51;
		if (itype==21 && jv==1) iq=52;
		if (itype==21 && iv==4 && jv==1) iq=53;
		if (itype==21 && iv==1 && jv==4) iq=53;

		if (itype==22 && iv==1) iq=51;
		if (itype==22 && jv==1) iq=52;
		if (itype==22 && iv==1 && jv==1) iq=53;
		if (itype==22 && iv==4 && jv==4) iq=53;


		if (itype==23 && (iv==1 || iv==4)) iq=51;
		if (itype==23 && (jv==1 || jv==4)) iq=52;
		if (itype==23 && (iv==1 && jv==1)) iq=53;
		if (itype==23 && (iv==1 && jv==4)) iq=53;
		if (itype==23 && (iv==4 && jv==1)) iq=53;
		if (itype==23 && (iv==4 && jv==4)) iq=53;

		if (itype==24 && iv==1) iq=51;
		if (itype==24 && iv==4 && jv==1) iq=53;
		if (itype==25 && jv==4) iq=52;
		if (itype==25 && iv==1 && jv==1) iq=53;
		if (itype==26 && iv==4) iq=52;
		if (itype==26 && iv==1 && jv==4) iq=53;
		if (itype==27 && jv==1) iq=52;
		if (itype==27 && iv==4 && jv==4) iq=53;

		if (itype==28 && iv==1) iq=51;
		if (itype==28 && iv==4 && jv==4) iq=53;
		if (itype==29 && jv==4) iq=52;
		if (itype==29 && iv==4 && jv==1) iq=53;
		if (itype==30 && iv==4) iq=52;
		if (itype==30 && iv==1 && jv==1) iq=53;
		if (itype==31 && jv==1) iq=52;
		if (itype==31 && iv==1 && jv==4) iq=53;


		if (itype==32 && iv==1) iq=51;
		if (itype==32 && iv==4 && jv==4) iq=53;
		if (itype==32 && iv==4 && jv==1) iq=53;
		if (itype==33 && jv==4) iq=52;
		if (itype==33 && iv==1 && jv==1) iq=53;
		if (itype==33 && iv==4 && jv==1) iq=53;
		if (itype==34 && iv==4) iq=52;
		if (itype==34 && iv==1 && jv==4) iq=53;
		if (itype==34 && iv==1 && jv==1) iq=53;
		if (itype==35 && jv==1) iq=52;
		if (itype==35 && iv==1 && jv==4) iq=53;
		if (itype==35 && iv==4 && jv==4) iq=53;

		if (itype==36 && iv==1 && jv==4) iq=53;
		if (itype==36 && iv==4 && jv==4) iq=53;
		if (itype==37 && iv==4 && jv==1) iq=53;
		if (itype==37 && iv==4 && jv==4) iq=53;
		if (itype==38 && iv==1 && jv==1) iq=53;
		if (itype==38 && iv==4 && jv==1) iq=53;
		if (itype==39 && iv==1 && jv==4) iq=53;
		if (itype==39 && iv==1 && jv==4) iq=53;


		if (itype==40 && iv==1 && jv==4) iq=53;
		if (itype==40 && iv==4 && jv==4) iq=53;
		if (itype==40 && iv==4 && jv==1) iq=53;
		if (itype==41 && iv==1 && jv==1) iq=53;
		if (itype==41 && iv==4 && jv==1) iq=53;
		if (itype==41 && iv==4 && jv==4) iq=53;
		if (itype==42 && iv==1 && jv==1) iq=53;
		if (itype==42 && iv==4 && jv==4) iq=53;
		if (itype==42 && iv==1 && jv==4) iq=53;
		if (itype==43 && iv==1 && jv==1) iq=53;
		if (itype==43 && iv==1 && jv==4) iq=53;
		if (itype==43 && iv==4 && jv==1) iq=53;

		if (itype==44 && ((iv==1 && jv==1) || (iv==4 && jv==1) ||
				  (iv==4 && jv==4) || (iv==1 && jv==4)) ) iq=53;

		if (itype==45 && ((iv==1 && jv==4) || (iv==4 && jv==1))) iq=53;
		if (itype==46 && ((iv==1 && jv==1) || (iv==4 && jv==4))) iq=53;

		mcount++;
		fprintf(fc,"%4d",iq);
		if (mcount>12) {
		    mcount=0;
		    fprintf(fc,"\n");
		}
	    }
	}
    }

    fprintf(fc,"\n/\n");

    fprintf(fc,"\nENDFIN\n");

}
