/*
 * ############################################################################
 * Name:      grd3d_prop_infill1_int.c
 * Author:    JRIV@statoil.com
 * Created:   2015-09-16
 * Updates:
 * ############################################################################
 * Look for lonely cells if certain value; will replace it by lookup in the
 * same layer, based on weigthing of similar Z value to cell
 *
 * Arguments:
 *     nx..nz           grid dimensions
 *     i1..k2           grid index window to work within
 *     p_zcorn_v        ZCORN array (pointer) of input
 *     p_actnum_v       ACTNUM array (pointer)
 *     p_sumdz_v        resulting DZ sum array (pointer)
 *     flip             use 1 or -1 for flipping vertical
 *     debug            debug/verbose flag
 *
 * Caveeats/issues:
 *     nothing known
 * ############################################################################
 */

#include <math.h>
#include "libxtg.h"
#include "libxtg_.h"


int grd3d_prop_infill1_int(
			   int nx,
			   int ny,
			   int nz,
			   int i1,
			   int i2,
			   int j1,
			   int j2,
			   int k1,
			   int k2,
			   double *p_coord_v,
			   double *p_zcorn_v,
			   int *p_actnum_v,
			   int *p_xxx_v,
			   int value,
			   int debug
			   )

{
    /* locals */
    int       i, j, k, ib, ibn, ii, jj, kk, iii, jjj, maxrad, icc, jcc, kcc;
    int       use_ib, tst_m, use_m, m, nrad, nc;
    double     *z_v, *zdiff, z0, throws[8], zdiffmin;
    int       *ic, rank[8], ibnn[8], nn, found, nrank, use_nn, nedge, istat;
    char      s[24]="grd3d_prop_infill1_int";

    xtgverbose(debug);
    xtg_speak(s,2,"Entering <grd3d_calc_sum_dz>");
    xtg_speak(s,3,"NX NY NZ: %d %d %d", nx, ny, nz);

    z_v=calloc(nx*ny*nz,sizeof(double));

    maxrad=4;

    zdiff=calloc((maxrad+1)*(maxrad+1),sizeof(double));
    ic=calloc((maxrad+1)*(maxrad+1),sizeof(int));

    if (z_v==NULL) {
	xtg_error(s,"STOP! Cannot allocate memory!");
    }

    grd3d_calc_z(nx,ny,nz,p_zcorn_v,z_v,debug);

    xtg_speak(s,2,"Searching per layer...");

    for (k = 1; k <= nz; k++) {

	xtg_speak(s,2,"Working with layer %d",k);

	for (j = j1; j <= j2; j++) {
	    for (i = i1; i <= i2; i++) {

		ib=x_ijk2ib(i,j,k,nx,ny,nz,0);

		if (p_xxx_v[ib] == value && p_actnum_v[ib] >= 1) {

		    /*
		     * Value found! ... now need to look at cells around
                     * within a radius
		     */

		    /* get the Z value */
		    z0=z_v[ib];
		    xtg_speak(s,2,"Found 'undef' cell %d %d %d with depth "
                              "%6.2f", i,j,k,z0);


		    /*
		     * Alt 1: Evaluate closest 8 cells (will hit in most
                     * of the time)
		     */

		    /* check frist if edge cell */
		    nedge=0;
		    if (i==1 || i==nx || j==1 || j==ny) {
			nedge=1;
		    }


		    if (nedge==0) {
			xtg_speak(s,2,"Trying Alt1 to find infill cells...");
			/* need the fault throws of the neighbour cells:
                           (for numbering, see grd3d_cell_faultthrows)*/

			istat = grd3d_cell_faultthrows(i, j, k, nx, ny, nz,
                                                       p_coord_v, p_zcorn_v,
                                                       p_actnum_v, throws, 0,
                                                       debug);
                        if (istat != 1) xtg_error(s, "Problem encountered "
                                                  "in %s", s);

			/* establish a rank based on fault throws */
			for (nn=0;nn<8;nn++) {
			    rank[nn]=2;
			    /* down rank the corner cells: */
			    if (nn==0 || nn==2 || nn==4 || nn==6) rank[nn]=1;

			    if (fabs(throws[nn])>0.001) rank[nn]=0;

			}

			if (i>1 && i<nx &&j>1 && j<ny) {
			    ibnn[0]=x_ijk2ib(i-1,j-1,k,nx,ny,nz,0);
			    ibnn[1]=x_ijk2ib(i-1,j,k,nx,ny,nz,0);
			    ibnn[2]=x_ijk2ib(i-1,j+1,k,nx,ny,nz,0);
			    ibnn[3]=x_ijk2ib(i,j+1,k,nx,ny,nz,0);
			    ibnn[4]=x_ijk2ib(i+1,j+1,k,nx,ny,nz,0);
			    ibnn[5]=x_ijk2ib(i+1,j,k,nx,ny,nz,0);
			    ibnn[6]=x_ijk2ib(i+1,j-1,k,nx,ny,nz,0);
			    ibnn[7]=x_ijk2ib(i,j-1,k,nx,ny,nz,0);
			}

			/* now check the depth difference */
			for (nn=0;nn<8;nn++) {
			    ibn=ibnn[nn];
			    if ((p_actnum_v[ibn] >= 1) && p_xxx_v[ibn] != value) {
				x_ib2ijk(ibn,&ii,&jj,&kk,nx,ny,nz,0);
				xtg_speak(s,2,"Evaluate cell %d %d %d with "
                                          "depth %6.2f", ii,jj,kk,z_v[ibn]);
				zdiff[nn]=fabs(z0-z_v[ibn]);
				xtg_speak(s,2," ... zdiff is %6.2f", zdiff[nn]);
			    }
			    /* modify the rank if the neighbour cell has
                             * the aka undef value */
			    if (p_xxx_v[ibn] == value) rank[nn]=-1;

			    xtg_speak(s,2,"Throw for neighbour %d is %6.2f, "
                                      "rank is %d",nn,throws[nn],rank[nn]);
			}

			/* now select based on rank and throw... */
			found=0;
			use_nn=0;
			zdiffmin=99999999;
			for (nrank=2; nrank>=0; nrank--) {
			    for (nn=0;nn<8;nn++) {
				if (rank[nn] == nrank) {
				    if (zdiff[nn]<zdiffmin) {
					zdiffmin=zdiff[nn];
					use_nn=nn;
					found=1;
				    }
				}
			    }

			    if (found==1) {
				ibn=ibnn[use_nn];
				x_ib2ijk(ibn,&ii,&jj,&kk,nx,ny,nz,0);
				xtg_speak(s,2,"Will use value from cell "
                                          "<%d %d %d> with rank %d and "
                                          "zdiff %6.2f",
					  ii,jj,kk,nrank,zdiff[use_nn]);

				p_xxx_v[ib]=p_xxx_v[ibn];
			    }

			    if (found==1) break;

			}
		    }
		    else{
			xtg_speak(s,2,"Edge cell; cannot use Alt. 1 method "
                                  "for <%d %d %d>",i,j,k);
			found=0;
		    }

		    /*
		     * Alt2: Evaluate a wider range radius
		     */

		    if (found==0) {
			xtg_speak(s,2,"Trying Alt2 for finding infill cells...");

			for (nrad=2; nrad<=maxrad; nrad++) {
			    nc=0; /* neighbour count */
			    xtg_speak(s,2,"Radius: %d", nrad);


			    for (jj=j-nrad;jj<=j+nrad;jj++) {
				for (ii=i-nrad;ii<=i+nrad;ii++) {
				    iii = ii;
				    jjj = jj;
				    if (ii<i1) iii=i1;
				    if (ii>i2) iii=i2;
				    if (jj<j1) jjj=j1;
				    if (jj>j2) jjj=j2;

				    ibn=x_ijk2ib(iii,jjj,k,nx,ny,nz,0);

				    if (ibn != ib && (p_actnum_v[ibn] >= 1) && p_xxx_v[ibn] != value) {
					xtg_speak(s,2,"Evaluate cell %d %d %d with depth %6.2f", ii,jj,k,z_v[ibn]);

					zdiff[nc]=fabs(z0-z_v[ibn]);
					xtg_speak(s,2," ... zdiff for nc %d is %6.2f", nc,zdiff[nc]);
					ic[nc]=ibn;
					nc++;
				    }
				}
			    }

			    if (nc>0) nc=nc-1;

			    /* for a given radius, zdiff is computed and stored as an array
			       as well as the corresponding ibn
			    */

			    xtg_speak(s,2,"NC is %d", nc);
			    if (nc<0) {
				xtg_speak(s,2,"Too few values for this radius <%d>; will extend radius",nrad);
			    }
			    else{
				use_m  = 0;
				tst_m  = 0;
				use_ib = ic[0];
				for (m=0; m<nc; m++) {
				    if (zdiff[m]<=zdiff[m+1]) {
					tst_m=m;
					xtg_speak(s,2,"ZDIFF for m = %d is smaller or equal", m);
				    }
				    else{
					tst_m=m+1;
					xtg_speak(s,2,"ZDIFF for m+1 = %d is smaller", m+1);
				    }

				    xtg_speak(s,2,"Test for m vs m+1: %d .. %d", m, m+1);
				    if (zdiff[tst_m] <= zdiff[use_m]) {
					use_m=tst_m;

				    }
				}

				use_ib=ic[use_m];

				/* a use_ib should exist... */
				p_xxx_v[ib]=p_xxx_v[use_ib];


				x_ib2ijk(use_ib,&icc,&jcc,&kcc,nx,ny,nz,0);

				if (debug>1) {
				    xtg_speak(s,2,"Value chosen for cell <%d %d %d> is <%d> from cell <%d %d %d>",
					      i,j,k, p_xxx_v[use_ib], icc, jcc, kcc);
				}
				break;
			    }
			    if (nrad==maxrad) {
				xtg_warn(s,1,"Did not find values to interpolate for cell <%d %d %d>",i,j,k);
				return(0);
			    }
			}
		    }
		}
	    }
	}
    }

    free(z_v);
    free(zdiff);
    free(ic);

    xtg_speak(s,2,"Exiting <grd3d_prop_infill1_int>");
    return(1);
}
