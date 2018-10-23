/*
 * ############################################################################
 * grd3d_adj_z_from_mapv3.c
 * ############################################################################
 * In version v3, based on v2, tries to use a COMPAT cell index to stiffen
 * actually not change, compdat cells. The compdat cells have 9 as value
 * ############################################################################
 */


#include "libxtg.h"
#include "libxtg_.h"

/*
 * ############################################################################
 * Use a map (correction map) to adjust zvalues in grid.
 * ############################################################################
 */

void grd3d_adj_z_from_mapv3 (
			     int nx,
			     int ny,
			     int nz,
			     double *p_coord_v,
			     double *p_zcorn_v,
			     int *p_actnum_v,
			     int mx,
			     int my,
			     double xmin,
			     double xstep,
			     double ymin,
			     double ystep,
			     double *p_map_v,
			     int   iflag,
			     int   flip,
			     int   debug
			     )

{
    /* locals */
    int     i, j, k, ib, ij, ic, kk;
    int     kaactfound, kbactfound, kb_use;
    double  x[5], y[5], zz;
    double  adj_zval, z1n = 0.0, z2n = 0.0;
    double  corner_v[24];
    double  *z_v;
    int     kafound, kbfound, kmode, debugx = 0;
    int     *iac;
    int     (*ka)[4], (*kb)[4];
    char    s[24]="grd3d_adj_z_from_mapv3";

    xtgverbose(debug);

    xtg_speak(s,2,"Entering <grd3d_adj_z_from_mapv3>");
    xtg_speak(s,3,"Using IFLAG: %d", iflag);

    xtg_speak(s,3,"NX NY NZ: %d %d %d", nx, ny, nz);
    xtg_speak(s,3,"MX and MY is: %d %d", mx, my);
    xtg_speak(s,3,"XMIN: %13.2f\n", xmin);


    /*
     * Must be sure that grid is consistent in z, and also has
     * a small separation for each cell-layer
     */

    grd3d_make_z_consistent(
			    nx,
			    ny,
			    nz,
			    p_zcorn_v,
			    p_actnum_v,
			    0.001,
			    debug
			    );




    /*
     * There is an "artifical" ACTNUM=9. These are well cells that
     * shall be locked
     * ref COMPDAT settings. The processing of this must be done in advance
     */

    ka=calloc(nx*ny*4, sizeof(int));
    kb=calloc(nx*ny*4, sizeof(int));
    z_v=calloc(nz+1, sizeof(double));

    iac=calloc(nx*ny, sizeof(int));

    xtg_speak(s,1,"Adjusting grid to map v3 - this may take some time...");

    if (iflag<=1) {
	kmode=0;
    }
    else{
	kmode=12;
    }


    /*
     * Scan and find k values, seen from top or bot, there a 4+4 k
     * values per cell
     * Note that IC counts from 0 to 3, which in normal meaning is
     * cellcorner 1..4
     */

    for (j = 1; j <= ny; j++) {
        xtg_speak(s,3,"Search - finished grid column %d of %d",j,ny);
	for (i = 1; i <= nx; i++) {
	    xtg_speak(s,4,"Finished grid row %d of %d",i,nx);

	    ij=x_ijk2ib(i,j,1,nx,ny,1,0);
	    /* use top cell Z values */
	    /* find first active cell, from top/bottom depending on method */

	    iac[ij]=0; /* assume inactive initially */
	    kafound=0;
	    kbfound=0;
	    for (k=1;k<=nz;k++) {
		ib=x_ijk2ib(i,j,k,nx,ny,nz+1,0);
		if (p_actnum_v[ib]>=1 && kafound==0) {
		    kafound=1;
		    iac[ij]=1;
		    for (ic=0;ic<4;ic++) ka[ij][ic]=k;


		    /* now transverse upwards */
		    for (kk=nz+1;kk>1;kk--) {
			ib=x_ijk2ib(i,j,kk-1,nx,ny,nz+1,0);
			if (p_actnum_v[ib]>=1 && kbfound==0) {
			    kbfound=1;
			    for (ic=0;ic<4;ic++) kb[ij][ic]=kk;

			}
		    }
		}
	    }

	}
    }


    /*
       Now look for COMPDAT cells. These are marked with ACTNUM=9, and
       they must stay between current top and bottom, for active columns.
       For kb, the most shallow finding must be used; vice versa for
       ka. In addition, neighbour corners are marked!
    */

    for (j = 1; j <= ny; j++) {
        xtg_speak(s,4,"Finished grid column %d of %d",j,ny);
	for (i = 1; i <= nx; i++) {
	    xtg_speak(s,4,"Finished grid row %d of %d",i,nx);

	    ij=x_ijk2ib(i,j,1,nx,ny,1,0);

	    kbactfound=0;
	    if (iflag<=1) {
		for (k=1;k<=nz;k++) {
		    ib=x_ijk2ib(i,j,k,nx,ny,nz+1,0);
		    if (p_actnum_v[ib]==9 && kbactfound==0) {

			kbactfound=1;

			xtg_speak(s,1,"COMPDAT cell found: %d %d K = %d",
				  i, j, k);


			for (ic=0;ic<4;ic++) {
			    if (k<kb[ij][ic]) kb[ij][ic]=k;
			}

			/*
			 * now the tricky part - the neighbour corners ...
			 *
			 * 3----4
			 * |    |  Cell corners. ic goes from, 0..3
			 * 1----2  (corner1=>0 etc)
			 *
			 */

			ij=x_ijk2ib(i+1,j,1,nx,ny,1,0);
			/* will return negative index if out of bounds*/
			if (ij>=0 && k<kb[ij][0]) kb[ij][0]=k;
			if (ij>=0 && k<kb[ij][2]) kb[ij][2]=k;
			/* ------------------------------*/
			ij=x_ijk2ib(i+1,j+1,1,nx,ny,1,0);
			if (ij>=0 && k<kb[ij][0]) kb[ij][0]=k;
			/* ------------------------------*/
			ij=x_ijk2ib(i,j+1,1,nx,ny,1,0);
			if (ij>=0 && k<kb[ij][0]) kb[ij][0]=k;
			if (ij>=0 && k<kb[ij][1]) kb[ij][1]=k;
			/* ------------------------------*/
			ij=x_ijk2ib(i-1,j+1,1,nx,ny,1,0);
			if (ij>=0 && k<kb[ij][1]) kb[ij][1]=k;
			/* ------------------------------*/
			ij=x_ijk2ib(i-1,j,1,nx,ny,1,0);
			if (ij>=0 && k<kb[ij][1]) kb[ij][1]=k;
			if (ij>=0 && k<kb[ij][3]) kb[ij][3]=k;
			/* ------------------------------*/
			ij=x_ijk2ib(i-1,j-1,1,nx,ny,1,0);
			if (ij>=0 && k<kb[ij][3]) kb[ij][3]=k;
			/* ------------------------------*/
			ij=x_ijk2ib(i,j-1,1,nx,ny,1,0);
			if (ij>=0 && k<kb[ij][2]) kb[ij][2]=k;
			if (ij>=0 && k<kb[ij][3]) kb[ij][3]=k;
			/* ------------------------------*/
			ij=x_ijk2ib(i+1,j-1,1,nx,ny,1,0);
			if (ij>=0 && k<kb[ij][2]) kb[ij][2]=k;


			ij=x_ijk2ib(i,j,1,nx,ny,1,0);    /* reset */


		    }

		}
	    }
	    else{

		kaactfound=0;
		for (k=nz+1;k>1;k--) {
		    ib=x_ijk2ib(i,j,k-1,nx,ny,nz+1,0);
		    if (p_actnum_v[ib]==9 && kaactfound==0) {

			kaactfound=1;

			xtg_speak(s,3,"COMPDAT cell found: %d %d K = %d",
				  i, j, k);


			for (ic=0;ic<4;ic++) {
			    if (k>ka[ij][ic]) ka[ij][ic]=k;
			}

			/* now the tricky part - the neighbour corners ... */


			ij=x_ijk2ib(i+1,j,1,nx,ny,1,0);
			/* will return negative index if out of bounds*/

			if (ij>=0 && k>ka[ij][0]) ka[ij][0]=k;
			if (ij>=0 && k>ka[ij][2]) ka[ij][2]=k;
			/* ------------------------------*/
			ij=x_ijk2ib(i+1,j+1,1,nx,ny,1,0);
			if (ij>=0 && k>ka[ij][0]) ka[ij][0]=k;
			/* ------------------------------*/
			ij=x_ijk2ib(i,j+1,1,nx,ny,1,0);
			if (ij>=0 && k>ka[ij][0]) ka[ij][0]=k;
			if (ij>=0 && k>ka[ij][1]) ka[ij][1]=k;
			/* ------------------------------*/
			ij=x_ijk2ib(i-1,j+1,1,nx,ny,1,0);
			if (ij>=0 && k>ka[ij][1]) ka[ij][1]=k;
			/* ------------------------------*/
			ij=x_ijk2ib(i-1,j,1,nx,ny,1,0);
			if (ij>=0 && k>ka[ij][1]) ka[ij][1]=k;
			if (ij>=0 && k>ka[ij][3]) ka[ij][3]=k;
			/* ------------------------------*/
			ij=x_ijk2ib(i-1,j-1,1,nx,ny,1,0);
			if (ij>=0 && k>ka[ij][3]) ka[ij][3]=k;
			/* ------------------------------*/
			ij=x_ijk2ib(i,j-1,1,nx,ny,1,0);
			if (ij>=0 && k>ka[ij][2]) ka[ij][2]=k;
			if (ij>=0 && k>ka[ij][3]) ka[ij][3]=k;
			/* ------------------------------*/
			ij=x_ijk2ib(i+1,j-1,1,nx,ny,1,0);
			if (ij>=0 && k>ka[ij][2]) ka[ij][2]=k;


			ij=x_ijk2ib(i,j,1,nx,ny,1,0);    /* reset */


		    }


		}
	    }


	}
    }

    /*
       =========================================================================
       Now every active cell column is adjusted, according to difference
       map. The indices were found for each corner earlier, and a general
       "stretch" routine simplifies work
    */

    if (iflag<=1) {
	for (j = 1; j <= ny; j++) {
	    xtg_speak(s,1,"Finished grid column %d of %d",j,ny);
	    for (i = 1; i <= nx; i++) {
		xtg_speak(s,4,"Finished grid row %d of %d",i,nx);

		ij=x_ijk2ib(i,j,1,nx,ny,1,0);

		/* each cell have 4 corners (0..3), which all may have
		   a different indexing */



		for (ic=0; ic<4; ic++) {


		    grd3d_corners(i,j,ka[ij][ic],nx,ny,nz,p_coord_v,
				  p_zcorn_v,corner_v,debug);

		    x[ic]=corner_v[ic*3+0+kmode];
		    /* e.g. y[3]=..[3*3+1+kmode]=..[10+kmode] */

		    y[ic]=corner_v[ic*3+1+kmode];

		    adj_zval=map_get_z_from_xy(x[ic],y[ic],mx,my,xstep,
					       ystep,xmin,ymin,p_map_v,debug);

		    if (adj_zval > UNDEF_LIMIT) {
			xtg_warn(s,1,"Grid is outside map. Cannot proceed!"
				 " Extend input map!");
			xtg_warn(s,1,"Cell is I=%d, J=%d K=%d IC=%d and"
				 " X=%f Y=%f",
				 i,j,ic,ka[ij][ic],x[ic],y[ic]);
		    }



		    for (k=1;k<=nz+1;k++) {
			ib=x_ijk2ib(i,j,k,nx,ny,nz+1,0);
			z_v[k-1]=p_zcorn_v[4*ib + 1*(ic+1) - 1];
			if (k==ka[ij][ic]) z1n=z_v[k-1] + adj_zval;
			if (k==kb[ij][ic] && k<nz+1) {
				z2n=z_v[k-1];
			}
			if (k==kb[ij][ic] && k==nz+1) {
			    if (iflag==0) z2n=z_v[k-1] + adj_zval;
			    if (iflag==1) z2n=z_v[k-1];
			}
		    }


		    /* adjustment should not "fold" grid */
		    if (z1n>z2n) z1n=z2n-FLOATEPS;

		    x_stretch_vector(z_v,nz+1,ka[ij][ic],kb[ij][ic],
				     z1n,z2n,debugx);

		    /* map modified value back */
		    for (k=1;k<=nz+1;k++) {
			ib=x_ijk2ib(i,j,k,nx,ny,nz+1,0);
			p_zcorn_v[4*ib + 1*(ic+1) - 1] = z_v[k-1];
		    }
		}
	    }
	}
    }
    if (iflag==2) {
	for (j = 1; j <= ny; j++) {
	    xtg_speak(s,1,"Finished grid column %d of %d",j,ny);
	    for (i = 1; i <= nx; i++) {
		xtg_speak(s,4,"Finished grid row %d of %d",i,nx);

		ij=x_ijk2ib(i,j,1,nx,ny,1,0);

		/* each cell have 4 corners (0..3), which all
		   may have a different indexing */

		for (ic=0; ic<4; ic++) {

		    /* since kb goes from nz+1, we need to take
		       the bottom of cell kb-1 */

		    kb_use=kb[ij][ic]-1;
		    if (kb_use<1) kb_use=1;

		    grd3d_corners(i,j,kb_use,nx,ny,nz,p_coord_v,
				  p_zcorn_v,corner_v,debug);

		    x[ic]=corner_v[ic*3+0+kmode];
		    /* e.g. y[22]=..[3*3+1+kmode]=..[10+12] */

		    y[ic]=corner_v[ic*3+1+kmode];
		    zz   =corner_v[ic*3+2+kmode];

		    adj_zval=map_get_z_from_xy(x[ic],y[ic],mx,my,xstep,
					       ystep,xmin,ymin,p_map_v,debug);

		    if (adj_zval > UNDEF_LIMIT) {
			xtg_warn(s,1,"Grid is outside map. Cannot proceed! "
				 "Extend input map!");
			xtg_warn(s,1,"Cell is I=%d, J=%d K=%d IC=%d and "
				 "X=%f Y=%f Z=%f",
				 i,j,kb_use,ic,x[ic],y[ic],zz);
			xtg_error(s,"STOP");
		    }

		    for (k=nz+1;k>=1;k--) {
			ib=x_ijk2ib(i,j,k,nx,ny,nz+1,0);
			z_v[k-1]=p_zcorn_v[4*ib + 1*(ic+1) - 1];
			if (k==kb[ij][ic]) z2n=z_v[k-1] + adj_zval;
			if (k==ka[ij][ic]) {
			    z1n=z_v[k-1]; /* top always fixed for iflag=2 */
			}
		    }


		    /* adjustment should not "fold" grid */
		    if (z1n>z2n) z2n=z1n+FLOATEPS;

		    x_stretch_vector(z_v,nz+1,ka[ij][ic],kb[ij][ic],
				     z1n,z2n,debugx);

		    /* map modified value back */
		    for (k=1;k<=nz+1;k++) {
			ib=x_ijk2ib(i,j,k,nx,ny,nz+1,0);
			p_zcorn_v[4*ib + 1*(ic+1) - 1] = z_v[k-1];
		    }
		}
	    }
	}
    }

    grd3d_make_z_consistent(
			    nx,
			    ny,
			    nz,
			    p_zcorn_v,
			    p_actnum_v,
			    0.00001,
			    debug
			    );


    free(iac);
    free(ka);
    free(kb);
    free(z_v);

    xtg_speak(s,1,"Adjusting grid to map v3 ... DONE!");
}
