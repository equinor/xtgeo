/*
 * ############################################################################
 * grd3d_adj_z_from_map.c
 * ############################################################################
 */

#include <math.h>
#include "libxtg.h"
#include "libxtg_.h"

/*
 * ############################################################################
 * Use a map (correction map) to adjust zvalues in grid.
 * ############################################################################
 */

void grd3d_adj_z_from_map (
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
			   double *p_grd2d_v,
			   double zconst,
			   int   iflag,
			   int   flip,
			   int   debug
			   )

{
    /* locals */
    int i, j, k, ib, ibm, ic, kk;
    int im, jm, imnode, jmnode, kstart, kstop, kdir, kmode;
    int imm, iwarn=0;
    double x[5], y[5], w[5];
    double  xpos1, ypos1, xpos2, ypos2, adj_zval=0.0;
    double corner_v[24];
    double  summ, adj, sum_w, z1, z2, z3;
    double  *zk;
    double  *p_sumdz_v;
    int    ib1, ib2, ib3;
    char   s[24]="grd3d_adj_z_from_map";

    xtgverbose(debug);

    xtg_speak(s,2,"Entering <grd3d_adj_z_from_map>");
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
			    zconst,
			    debug
			    );


    p_sumdz_v=calloc(nx*ny*nz,4);
    grd3d_calc_sum_dz(
		      nx,
		      ny,
		      nz,
		      p_zcorn_v,
		      p_actnum_v,
		      p_sumdz_v,
		      flip,
		      debug
		      );



    /*
     * Loop grid top values. For each depth point, find the average map value
     * and use that one.
     */


    xtg_speak(s,2,"Adjusting grid to map - this may take some time...");
    for (j = 1; j <= ny; j++) {
        xtg_speak(s,2,"Finished column %d of %d",j,ny);
	for (i = 1; i <= nx; i++) {

	    /* use top cell Z values */
	    /* find first active cell, from top/bottom depending on method */

	    /* iflag==0 or iflag==1: work from top */
	    kstart=0;
	    kstop=0;
	    if (iflag<=1) {
		for (k=1;k<=nz;k++) {
		    ib=x_ijk2ib(i,j,k,nx,ny,nz+1,0);
		    if (p_actnum_v[ib]==1) {
			kstart=k;
			kdir=1;
			kmode=0;
			grd3d_corners(i,j,k,nx,ny,nz,p_coord_v,
				      p_zcorn_v,corner_v,debug);
			for (kk=nz;kk>=k;kk--) {
			    ib=x_ijk2ib(i,j,kk,nx,ny,nz+1,0);
			    if (p_actnum_v[ib]==1) {
				kstop=kk;
				break;
			    }
			}
			break;
		    }
		}
	    }
	    else{

		for (k=nz;k>=1;k--) {
		    ib=x_ijk2ib(i,j,k,nx,ny,nz+1,0);
		    if (p_actnum_v[ib]==1) {
			kstart=k;
			kdir=-1;
			kmode=12;
			grd3d_corners(i,j,k,nx,ny,nz,p_coord_v,
				      p_zcorn_v,corner_v,debug);
			for (kk=1;kk<=k;kk++) {
			    ib=x_ijk2ib(i,j,kk,nx,ny,nz+1,0);
			    if (p_actnum_v[ib]==1) {
				kstop=kk;
				break;
			    }
			}
			break;
		    }
		}
	    }


	    /* The full cell column may be inactive */
	    if (kstart==0 && kstop==0) goto NEXTLOOP;

	    xtg_speak(s,4,"Column is I=%d and J=%d, KSTART=%d KSTOP=%d",
		      i,j,kstart,kstop);

	    /*
	     * Look at the corners that should be compared with map
	     * If kmode==0, top points are used, if kmode=12,
	     * bottom nodes are used
	     */


	    x[1]=corner_v[0+kmode];
	    y[1]=corner_v[1+kmode];
	    x[2]=corner_v[3+kmode];
	    y[2]=corner_v[4+kmode];
	    x[3]=corner_v[6+kmode];
	    y[3]=corner_v[7+kmode];
	    x[4]=corner_v[9+kmode];
	    y[4]=corner_v[10+kmode];


	    /* scan map */
	    xtg_speak(s,4,"Scanning map...");


	    for (ic=1;ic<=4;ic++) {
		for (jm=1;jm<my;jm++) {
		    for (im=1;im<mx;im++) {
			xpos1=xmin + (im-1)*xstep;
			ypos1=ymin + (jm-1)*ystep;
			xpos2=xmin + (im)*xstep;
			ypos2=ymin + (jm)*ystep;

			if (xpos1 < x[ic] && xpos2 >= x[ic] &&
			    ypos1 < y[ic] && ypos2 >= y[ic]) {

			    /*
			     * OK, we are inside mapnodes. Now I need a simple weight
			     * |---------------|
			     * |               | Use a quadratic inverse formula
			     * |           *   | for weights w[]
			     * |---------------|
			     */

			    w[1]=pow((x[ic]-xpos1),2) + pow((y[ic]-ypos1),2);
			    w[2]=pow((x[ic]-xpos2),2) + pow((y[ic]-ypos1),2);
			    w[3]=pow((x[ic]-xpos1),2) + pow((y[ic]-ypos2),2);
			    w[4]=pow((x[ic]-xpos2),2) + pow((y[ic]-ypos2),2);
			    summ=0.0;
			    for (imm=1; imm<=4; imm++) {
				if (w[imm]>0.00001) {
				    w[imm]=1.0/w[imm];
				}
				else{
				    w[imm]=10000000.0;
				}
				summ=summ+w[imm];
			    }
			    /* this should scale weights so that summ w[i] = 1.0 */
			    for (imm=1; imm<=4; imm++) {
				w[imm]=w[imm]/summ;
			    }


			    /*
			     * Find the adjustment for this node, weighted on distance
			     * from map node...
			     * UNDEF map nodes are treated as zero
			     */
			    adj_zval=0.0;
			    sum_w=0.0;
			    for (imm=1; imm<=4; imm++) {
				imnode=im;
				jmnode=jm;


				if (imm==2 || imm==4) imnode=im+1;
				if (imm==3 || imm==4) jmnode=jm+1;

				ibm=x_ijk2ib(imnode,jmnode,1,mx,my,1,0);
				adj=p_grd2d_v[ibm];
				if (p_grd2d_v[ibm] > UNDEF_MAP_LIMIT) adj=0.0;

				adj_zval=adj_zval + w[imm]*adj;
				sum_w=sum_w+w[imm];
			    }



			    /*
			     * If iflag=1 or 2, then it is required that bottom or top stays
			     * untouched. Hence, correction must be less that thickness of grid
			     * in cases where the correction makes a thinner than total grid thickness
			     */
			    if (iflag > 0) {
				ib=x_ijk2ib(i,j,kstart,nx,ny,nz+1,0); /* use k=kstart or
								       * any other active */

				if (iflag == 1) {
				    if (adj_zval >= (p_sumdz_v[ib]-0.01)) {
					adj_zval=(p_sumdz_v[ib]-0.01); /* allow 1cm remaing thickness */
					if (iwarn!=1) {
					    iwarn=-1;
					}
				    }
				}
				if (iflag == 2) {
				    /* a big negative adj_zval may cause problems */
				    if ((-1*adj_zval) >= (p_sumdz_v[ib]-0.01)) {
					adj_zval=-1*(p_sumdz_v[ib]-0.01); /* allow 1cm remaing thickness */
					if (iwarn!=1) {
					    iwarn=-2;
					}
				    }
				}
				if (iwarn<0) {
				    xtg_warn(s,1,"Adjustment is greater than total grid thicness.");
				    xtg_warn(s,1,"Adjustvalue (map) is reset to grid thickness at");
				    xtg_warn(s,1,"this point. This warning is only given once.");
				    iwarn=1;
				}
			    }

			    goto FOUND;
			}
		    }
		}
	    FOUND:


		/* Mapnodes should now be present. Adjust all values along pillar: */
		/* for (k = kstart; k <= kstop; k+=kdir) { */
		/* Adjust all cells in column the same amount */
		if (iflag==0) {
		    for (k = 1; k <= nz+1; k+=1) {
			ib=x_ijk2ib(i,j,k,nx,ny,nz+1,0);
			p_zcorn_v[4*ib + 1*ic - 1] =
			    p_zcorn_v[4*ib + 1*ic - 1] + adj_zval;
		    }
		}
		/* Adjust from top and gradually to bottom */
		else if (iflag==1) {
		    zk=calloc(nz+2,4); /* makes xtra nz+2 since we start counting from 1 */
		    ib1=x_ijk2ib(i,j,kstart,nx,ny,nz+1,0);
		    ib3=x_ijk2ib(i,j,kstop+1,nx,ny,nz+1,0);
		    z1=p_zcorn_v[4*ib1 + 1*ic - 1];
		    z3=p_zcorn_v[4*ib3 + 1*ic - 1];


		    if (fabs(z3-z1)>0) {

			for (kk = kstart; kk <= kstop+1; kk+=1) {
			    ib2=x_ijk2ib(i,j,kk,nx,ny,nz+1,0);
			    z2=p_zcorn_v[4*ib2 + 1*ic - 1];
			    zk[kk]=adj_zval - adj_zval*(z2-z1)/(z3-z1);
			    if (fabs(zk[kk]) > 100) {
				xtg_warn(s,3,"zk for k=%d in cell %d %d is %f",kk,i,j,zk[kk]);
			    }
			}


			for (k = 1; k <= nz+1; k+=1) {
			    ib=x_ijk2ib(i,j,k,nx,ny,nz+1,0);

			    if (k<=kstart) {
				p_zcorn_v[4*ib + 1*ic - 1] =
				    p_zcorn_v[4*ib + 1*ic - 1] + adj_zval;
			    }
			    else if (k>kstart && k <= kstop+1) {
				p_zcorn_v[4*ib + 1*ic - 1] =
				    p_zcorn_v[4*ib + 1*ic - 1] + zk[k];
			    }
			}

		    }
		    free(zk);
		}
		else if (iflag==2) {

		    /* Adjust from bottom and gradually to top */
		    zk=calloc(nz+2,4);
		    ib1=x_ijk2ib(i,j,kstart+1,nx,ny,nz+1,0);
		    ib3=x_ijk2ib(i,j,kstop,nx,ny,nz+1,0);
		    z1=p_zcorn_v[4*ib1 + 1*ic - 1];
		    z3=p_zcorn_v[4*ib3 + 1*ic - 1];


		    if (fabs(z3-z1)>0) {

			for (kk = kstart+1; kk >= kstop; kk+=kdir) {
			    ib2=x_ijk2ib(i,j,kk,nx,ny,nz+1,0);
			    z2=p_zcorn_v[4*ib2 + 1*ic - 1];
			    zk[kk]=adj_zval - adj_zval*(z1-z2)/(z1-z3);
			}

			for (k = nz+1; k >= 1; k--) {
			    ib=x_ijk2ib(i,j,k,nx,ny,nz+1,0);
			    if (k>(kstart+1)) {
				p_zcorn_v[4*ib + 1*ic - 1] =
				    p_zcorn_v[4*ib + 1*ic - 1] + adj_zval;
			    }
			    else if (k<=(kstart+1) && k >= kstop) {
				p_zcorn_v[4*ib + 1*ic - 1] =
				    p_zcorn_v[4*ib + 1*ic - 1] + zk[k];
			    }


			    }

		    }

		    free(zk);
		}


	    }

	    xtg_speak(s,4,"Scanning map... DONE!");

	NEXTLOOP:
	    xtg_speak(s,4,"Finished with cell I = %d J = %d",i,j);
	}

    }
    xtg_speak(s,2,"Freeing mem...");
    free(p_sumdz_v);

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
			    zconst,
			    debug
			    );

    xtg_speak(s,2,"Adjusting grid to map ... DONE!");
    xtg_speak(s,2,"Exiting <grd3d_adj_z_from_map>");
}
