/*
 * ############################################################################
 * grd3d_convert_hybrid.c
 * ############################################################################
 */


#include "libxtg.h"
#include "libxtg_.h"

/*
 * ############################################################################
 * Will convert an oridnary grid to an hybrid grid; being horizontal between
 * 2 levels, and then follow the layering.
 * ############################################################################
 */

void grd3d_convert_hybrid (
			   int    nx,
			   int    ny,
			   int    nz,
			   double *p_coord_v,
			   double *p_zcorn_v,
			   int    *p_actnum_v,
			   int    nzhyb,
			   double *p_zcornhyb_v,
			   int    *p_actnumhyb_v,
			   int    *p_num_act,
			   double  toplevel,
			   double  botlevel,
			   int    ndiv,
			   int    debug
			   )

{
    /* locals */
    int     i, j, k, n, ic, ibp, ibh, inp=0, inh=0, khyb, mi, iflagt, iflagb;
    double  z1, dz, zsum, ztop, zbot, zhyb, zsumh;
    char    sub[24]="grd3d_convert_hybrid";

    xtgverbose(debug);

    xtg_speak(sub,1,"Entering haha <grd3d_convert_hybrid>");


    /* thickness in horizontal section */
    dz=(botlevel-toplevel)/ndiv;

    for (j = 1; j <= ny; j++) {
	mi=j % 10; /* modulus */
 	if (mi==0) xtg_speak(sub,1,"Finished column %d of %d",j,ny);
	if (mi!=0 && j==ny) xtg_speak(sub,1,"Finished column %d of %d",j,ny);
	for (i = 1; i <= nx; i++) {
	    iflagt=1; iflagb=1; ztop=UNDEF; zbot=-1*UNDEF;
	    /* first do the top-down truncation, collecting all at toplevel */
	    for (k = 1; k <= nz+1; k++) {
		ibp=x_ijk2ib(i,j,k,nx,ny,nz+1,0);
		ibh=x_ijk2ib(i,j,k,nx,ny,nzhyb+1,0);
		/* do for all corners */
		zsum=0.0;
		for (ic=1;ic<=4;ic++) {
		    z1=p_zcorn_v[4*ibp + 1*ic - 1];
		    if (z1>toplevel) {
			p_zcornhyb_v[4*ibh + 1*ic - 1]=toplevel;
		    }
		    else{
			p_zcornhyb_v[4*ibh + 1*ic - 1]=z1;
		    }
		    /* store avg top depth; will be used to truncate later if active*/
		    zsum=zsum+z1;

		}
		/* now store top depth in input grid */
		if (k<=nz) {
		    if (p_actnum_v[ibp]==1 && iflagt==1) {
			ztop=zsum/4.0;
			iflagt=0;
		    }
		    p_actnumhyb_v[ibh] = p_actnum_v[ibp];
		    }
		}



	    /* now doing it the other way (from bottom) */
	    khyb=nzhyb+1;
	    for (k = nz+1; k >= 1; k--) {
		ibp=x_ijk2ib(i,j,k,nx,ny,nz+1,0);
		ibh=x_ijk2ib(i,j,khyb,nx,ny,nzhyb+1,0);

		/* in terms of active cells index, layer k _bottom_ shall refer to cell k-1 */
		if (k>1) {
		    inp=x_ijk2ib(i,j,k-1,nx,ny,nz+1,0);
		    inh=x_ijk2ib(i,j,khyb-1,nx,ny,nzhyb+1,0);
		}

		/* do for all corners */
		zsum=0.0;
		for (ic=1;ic<=4;ic++) {
		    z1=p_zcorn_v[4*ibp + 1*ic - 1];
		    if (z1<botlevel) {
			p_zcornhyb_v[4*ibh + 1*ic - 1]=botlevel;
		    }
		    else{
			p_zcornhyb_v[4*ibh + 1*ic - 1]=z1;
		    }
		    /* store avg bot depth; will be used to truncate later if active*/
		    zsum=zsum+z1;
		}
		/* now bot depth from input grid */
		if (k>1) {
		    if (p_actnum_v[inp]==1 && iflagb==1) {
			zbot=zsum/4.0;
			iflagb=0;
		    }
		    p_actnumhyb_v[inh] = p_actnum_v[inp];
		}
		khyb--;
	    }


	    /* now filling the intermediate */
	    n=0;
	    for (k = nz+1; k <= nz+1+ndiv-1; k++) {
		ibh=x_ijk2ib(i,j,k,nx,ny,nzhyb+1,0);
		/* do for all corners */
		if (k>nz+1) {
		    n++;
		    for (ic=1;ic<=4;ic++) {
			p_zcornhyb_v[4*ibh + 1*ic - 1]=toplevel+n*dz;
		    }
		}
		p_actnumhyb_v[ibh]=1;
	    }

	    /* truncate - ensure same volume, first from top, eval by cell centre */
	    for (k = 1; k <= nzhyb; k++) {
		zsumh=0.0;
		ibh=x_ijk2ib(i,j,k,nx,ny,nzhyb+1,0);
		inh=x_ijk2ib(i,j,k+1,nx,ny,nzhyb+1,0);
		/* do for all corners */
		for (ic=1;ic<=4;ic++) {
		    zsumh=zsumh+p_zcornhyb_v[4*ibh + 1*ic - 1];
		}
		for (ic=1;ic<=4;ic++) {
		    zsumh=zsumh+p_zcornhyb_v[4*inh + 1*ic - 1];
		}
		zhyb=0.125*zsumh; /* cell center */

		/* debug */
		//if (i==1 &&j==12) {
		//    printf("ztop zhyb khyb actnumhyb   %8.2f   %8.2f   %6d %6d\n",ztop,zhyb,k,p_actnumhyb_v[ibh]);
		//}

		if (p_actnumhyb_v[ibh]==1 && zhyb<ztop) {
		    p_actnumhyb_v[ibh]=0;
		}
	    }


	    /* truncate - ensure same volume, now from bot, eval by cell centre */
	    for (k = nzhyb+1; k > 1; k--) {
		zsumh=0.0;
		ibh=x_ijk2ib(i,j,k,nx,ny,nzhyb+1,0);
		inh=x_ijk2ib(i,j,k-1,nx,ny,nzhyb+1,0);
		/* do for all corners */
		for (ic=1;ic<=4;ic++) {
		    zsumh=zsumh+p_zcornhyb_v[4*ibh + 1*ic - 1];
		}
		for (ic=1;ic<=4;ic++) {
		    zsumh=zsumh+p_zcornhyb_v[4*inh + 1*ic - 1];
		}
		zhyb=0.125*zsumh;

		/* debug */
		//if (i==1 &&j==12) {
		//    printf("zbot zhyb khyb actnumhyb   %8.2f   %8.2f   %6d %6d\n",zbot,zhyb,k,p_actnumhyb_v[inh]);
		//}

		if (p_actnumhyb_v[inh]==1 && zhyb>zbot) {
		    p_actnumhyb_v[inh]=0;
		}
	    }
	}

    }


    xtg_speak(sub,2,"Exit from <grd3d_create_hybrid>");

}
