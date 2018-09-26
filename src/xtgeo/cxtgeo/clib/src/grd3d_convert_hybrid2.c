/*
 * ############################################################################
 * grd3d_convert_hybrid2.c
 * In this version, it can work within a region
 * ############################################################################
 */


#include "libxtg.h"
#include "libxtg_.h"

/*
 * ############################################################################
 * Will convert an ordinary grid to an hybrid grid; being horizontal between
 * 2 levels, and then follow the layering. This version uses regions. Regions
 * are horizontal only; that means it is not restricted to a particular zone;
 * this limitation will be handles in next version. Note that regions restricted
 * to a zone will be treated as it is valid for the full block; hence one region
 * cell in layer 66 will indicate that the full region block (XY) is hybrid.
 * ############################################################################
 */

void grd3d_convert_hybrid2 (
			    int   nx,
			    int   ny,
			    int   nz,
			    double *p_coord_v,
			    double *p_zcorn_v,
			    int   *p_actnum_v,
			    int   nzhyb,
			    double *p_zcornhyb_v,
			    int   *p_actnumhyb_v,
			    int   *p_num_act,
			    double toplevel,
			    double botlevel,
			    int   ndiv,
			    double *p_region_v,
			    int   region,
			    int   debug
			    )

{
    /* locals */
    int    i, ib, j, k, n, ic, ibp, ibh, inp=0, inh=0, khyb, mi;
    int    iflagt, iflagb, iflagr, actual_region=0;
    double  usetoplevel1, usetoplevel2, usetoplevel3, usetoplevel4;
    double  usebotlevel1, usebotlevel2, usebotlevel3, usebotlevel4;
    double  z1, z2, z3, z4, dz, zsum, ztop, zbot, zhyb, zsumh, usedz;
    char sub[24]="grd3d_convert_hybrid2";

    xtgverbose(debug);

    xtg_speak(sub,1,"Entering routine <grd3d_convert_hybrid2>");

    usetoplevel1=0.0; usetoplevel2=0.0; usetoplevel3=0.0; usetoplevel4=0.0;
    usebotlevel1=0.0; usebotlevel2=0.0; usebotlevel3=0.0; usebotlevel4=0.0;


    /* thickness in horizontal section */
    dz=(botlevel-toplevel)/ndiv;
    usedz=dz;

    xtg_speak(sub,2,"Dimens NX NY NZ NZHYB %d %d %d %d", nx, ny, nz, nzhyb);
    xtg_speak(sub,2,"DZ computed %f", usedz);

    for (j = 1; j <= ny; j++) {
	mi=j % 10; /* modulus */
 	if (mi==0) xtg_speak(sub,1,"Finished column %d of %d",j,ny);
	if (mi!=0 && j==ny) xtg_speak(sub,1,"Finished column %d of %d",j,ny);
	for (i = 1; i <= nx; i++) {


	    iflagt=1; iflagb=1; ztop=UNDEF; zbot=-1*UNDEF;

	    /* first need to scan K column to see if any hybrid level within the column;
	     * if hybrid region is found, the toplevel is used; otherwise the actual cells
	     * is used: however as going down, it will end with tha last bottom layer
	     */

	    iflagr=0;
	    usedz=dz;


	    for (k = 1; k <= nz+1; k++) {


		ibp=x_ijk2ib(i,j,k,nx,ny,nz+1,0);

                if (k <= nz) {
                    ib = x_ijk2ib(i,j,k,nx,ny,nz,0);
                    actual_region = x_nint(p_region_v[ib]);
                }

		/* this will end with last bottom layer unless hybrid region is found...*/
		if (actual_region != region && iflagr == 0){
		    usetoplevel1=p_zcorn_v[4*ibp + 1*1 - 1];
		    usetoplevel2=p_zcorn_v[4*ibp + 1*2 - 1];
		    usetoplevel3=p_zcorn_v[4*ibp + 1*3 - 1];
		    usetoplevel4=p_zcorn_v[4*ibp + 1*4 - 1];
		}

		if (actual_region == region){
		    iflagr=1;
		    usetoplevel1=toplevel;
		    usetoplevel2=toplevel;
		    usetoplevel3=toplevel;
		    usetoplevel4=toplevel;
		}

	    }


	    /* first do the top-down truncation, collecting all at toplevel (which is either
	     * the hybrid top OR the last layer)
	     */

	    for (k = 1; k <= nz+1; k++) {
		ibp=x_ijk2ib(i,j,k,nx,ny,nz+1,0);


		ibh=x_ijk2ib(i,j,k,nx,ny,nzhyb+1,0);
		/* do for all corners */
		zsum=0.0;


		/* CORNER 1 */
		z1=p_zcorn_v[4*ibp + 1*1 - 1];
		if (z1>usetoplevel1) {
		    p_zcornhyb_v[4*ibh + 1*1 - 1]=usetoplevel1;
		}
		else{
		    p_zcornhyb_v[4*ibh + 1*1 - 1]=z1;
		}
		/* store avg top depth; will be used to truncate later if active*/
		zsum=zsum+z1;

		/* CORNER 2 */
		z2=p_zcorn_v[4*ibp + 1*2 - 1];
		if (z2>usetoplevel2) {
		    p_zcornhyb_v[4*ibh + 1*2 - 1]=usetoplevel2;
		}
		else{
		    p_zcornhyb_v[4*ibh + 1*2 - 1]=z2;
		}
		/* store avg top depth; will be used to truncate later if active*/
		zsum=zsum+z2;


		/* CORNER 3 */
		z3=p_zcorn_v[4*ibp + 1*3 - 1];
		if (z3>usetoplevel3) {
		    p_zcornhyb_v[4*ibh + 1*3 - 1]=usetoplevel3;
		}
		else{
		    p_zcornhyb_v[4*ibh + 1*3 - 1]=z3;
		}
		/* store avg top depth; will be used to truncate later if active*/
		zsum=zsum+z3;


		/* CORNER 4 */
		z4=p_zcorn_v[4*ibp + 1*4 - 1];
		if (z4>usetoplevel4) {
		    p_zcornhyb_v[4*ibh + 1*4 - 1]=usetoplevel4;
		}
		else{
		    p_zcornhyb_v[4*ibh + 1*4 - 1]=z4;
		}
		/* store avg top depth; will be used to truncate later if active*/
		zsum=zsum+z4;




		/* now store top depth in input grid; it will be needed for later usage */
		if (k<=nz) {
		    if (p_actnum_v[ibp]==1 && iflagt==1) {
			ztop=zsum/4.0;
			iflagt=0;
		    }


		    /* inherit the ACTNUM */
		    p_actnumhyb_v[ibh] = p_actnum_v[ibp];
		}
	    }








	    /* now doing it the other way (from bottom), but here the botlevel shall be either
	     * hybdir bottom OR the base layer, which shall be similar to the one used when looping
	     * top down (see above)
	     */

	    zsum=0;
	    khyb=nzhyb+1;
	    for (k = nz+1; k >= 1; k--) {

		if (iflagr==0){
		    usebotlevel1=usetoplevel1;
		    usebotlevel2=usetoplevel2;
		    usebotlevel3=usetoplevel3;
		    usebotlevel4=usetoplevel4;
		}
		else{
		    usebotlevel1=botlevel;
		    usebotlevel2=botlevel;
		    usebotlevel3=botlevel;
		    usebotlevel4=botlevel;
		}

		ibp=x_ijk2ib(i,j,k,nx,ny,nz+1,0);

		ibh=x_ijk2ib(i,j,khyb,nx,ny,nzhyb+1,0);

		/* in terms of active cells index, layer k _bottom_ shall refer to cell k-1 */
		if (k>1) {
		    inp=x_ijk2ib(i,j,k-1,nx,ny,nz+1,0);
		    inh=x_ijk2ib(i,j,khyb-1,nx,ny,nzhyb+1,0);
		}


		/* CORNER 1 */
		z1=p_zcorn_v[4*ibp + 1*1 - 1];
		if (z1<usebotlevel1) {
		    p_zcornhyb_v[4*ibh + 1*1 - 1]=usebotlevel1;
		}
		else{
		    p_zcornhyb_v[4*ibh + 1*1 - 1]=z1;
		}
		/* store avg bot depth; will be used to truncate later if active*/
		zsum=zsum+z1;


		/* CORNER 2 */
		z2=p_zcorn_v[4*ibp + 1*2 - 1];
		if (z2<usebotlevel2) {
		    p_zcornhyb_v[4*ibh + 1*2 - 1]=usebotlevel2;
		}
		else{
		    p_zcornhyb_v[4*ibh + 1*2 - 1]=z2;
		}
		/* store avg bot depth; will be used to truncate later if active*/
		zsum=zsum+z2;


		/* CORNER 3 */
		z3=p_zcorn_v[4*ibp + 1*3 - 1];
		if (z3<usebotlevel3) {
		    p_zcornhyb_v[4*ibh + 1*3 - 1]=usebotlevel3;
		}
		else{
		    p_zcornhyb_v[4*ibh + 1*3 - 1]=z3;
		}
		/* store avg bot depth; will be used to truncate later if active*/
		zsum=zsum+z3;


		/* CORNER 4 */
		z4=p_zcorn_v[4*ibp + 1*4 - 1];
		if (z4<usebotlevel4) {
		    p_zcornhyb_v[4*ibh + 1*4 - 1]=usebotlevel4;
		}
		else{
		    p_zcornhyb_v[4*ibh + 1*4 - 1]=z4;
		}
		/* store avg bot depth; will be used to truncate later if active*/
		zsum=zsum+z4;



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
	    if (iflagr==0) {
		usedz=0.0;
	    }

	    for (k = nz+1; k <= nz+1+ndiv-1; k++) {
		ibh=x_ijk2ib(i,j,k,nx,ny,nzhyb+1,0);
		/* do for all corners */
		if (k>nz+1) {
		    n++;

		    p_zcornhyb_v[4*ibh + 1*1 - 1]=usetoplevel1+n*usedz;
		    p_zcornhyb_v[4*ibh + 1*2 - 1]=usetoplevel2+n*usedz;
		    p_zcornhyb_v[4*ibh + 1*3 - 1]=usetoplevel3+n*usedz;
		    p_zcornhyb_v[4*ibh + 1*4 - 1]=usetoplevel4+n*usedz;

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


    xtg_speak(sub,2,"Exit from <grd3d_create_hybrid2>");

}
