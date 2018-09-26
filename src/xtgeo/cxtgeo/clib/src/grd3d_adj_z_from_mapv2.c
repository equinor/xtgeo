/*
 * ############################################################################
 * grd3d_adj_z_from_mapv2.c
 * ############################################################################
 * In version v2, a faster and more stable algorithm is tried. This could
 * replace the orignal if it works
 * The idea is: faster lookup of map value, and better strucuring. The orig
 * will also trigger a segmentation error in many cases (espec with option 2)
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

void grd3d_adj_z_from_mapv2 (
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
    int     i, j, k, ib, ic, kk;
    int     kstart, kstop, kdir, kmode, kactive, iwarn=0;
    double  x[5], y[5];
    double  adj_zval;
    double  corner_v[24];
    double  z1, z2, z3;
    double  *zk;

    double  *p_sumdz_v;
    int     ib1, ib2, ib3;
    char    s[24]="grd3d_adj_z_from_mapv2";

    xtgverbose(debug);

    xtg_speak(s,2,"Entering <grd3d_adj_z_from_mapv2>");
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


    xtg_speak(s,1,"Adjusting grid to map - this may take some time...");

    for (j = 1; j <= ny; j++) {
        xtg_speak(s,2,"Finished grid column %d of %d",j,ny);
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
	    kactive=1;
	    if (kstart==0 && kstop==0) kactive=0;

	    if (kactive==1) {

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

		    // get the map depth of this corner
		    adj_zval=map_get_z_from_xy(x[ic],y[ic],mx,my,xstep,ystep,xmin,ymin,p_map_v,debug);
		    if (adj_zval > UNDEF_LIMIT) {
			xtg_error(s,"Grid is outside map. Cannot proceed! Extend input map!");
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


		    /* Adjust all values along pillar: */
		    /* for (k = kstart; k <= kstop; k+=kdir) { */
		    if (iflag==0) {
			for (k = 1; k <= nz+1; k+=1) {
			    ib=x_ijk2ib(i,j,k,nx,ny,nz+1,0);
			    p_zcorn_v[4*ib + 1*ic - 1] =
				p_zcorn_v[4*ib + 1*ic - 1] + adj_zval;
			}
		    }

		    /* Adjust from top and gradually to bottom */
		    else if (iflag==1) {
			zk=calloc(nz+2, sizeof(double));
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
			zk=calloc(nz+2, sizeof(double));
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
	    }
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
			    0.001,
			    debug
			    );

    xtg_speak(s,1,"Adjusting grid to map ... DONE!");
    xtg_speak(s,2,"Exiting <grd3d_adj_z_from_map>");
}
