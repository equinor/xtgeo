/*
 ******************************************************************************
 *
 * Inactivate (actnum) inside or outside a closed polygon
 *
 ******************************************************************************
 */

#include "libxtg.h"
#include "libxtg_.h"

/*
 ******************************************************************************
 *
 * NAME:
 *    grd3d_inact_outs_pol.c
 *
 * AUTHOR(S):
 *    Jan C. Rivenaes
 *
 * DESCRIPTION:
 *    Check all cells in a layer or subgrid, and set active to zero if outside
 *    or inside
 *
 *    The algorithm is to see if grid nodes lies inside some of the polygons.
 *    If not, an undef value is given. If already undef, then value is kept.
 *
 *    Note, polygons must have a 999 flag line for each new polygon
 *
 * ARGUMENTS:
 *    p_xp_v ...     i     Polygons vectors
 *    npx, npy       i     number of points in the polygon
 *    nx, ny, nz     i     Dimensions for 3D grid
 *    coord.. zcorn  i     Grid coordinates
 *    actnum        i/o    ACTNUM array
 *    nn             i     Length of array (for SWIG)
 *    k1, k2         i     K range min max
 *    option         i     0 inactivate inside, 1 inactivate outside
 *    debug          i     Debug level
 *
 * RETURNS:
 *    0 if all is OK, 1 if there are polygons that have problems.
 *    Result ACNUM is updated
 *
 * TODO/ISSUES/BUGS:
 *    Todo: The algorithm is straightforward and hence a bit slow...
 *
 * LICENCE:
 *    Statoil property
 ******************************************************************************
 */

/* the Python version; skip subgrids; use K ranges, multiple polygons */
int grd3d_inact_outside_pol(
                            double *p_xp_v,
                            long   npx,
                            double *p_yp_v,
                            long   npy,
                            int    nx,
                            int    ny,
                            int    nz,
                            double *p_coord_v,
                            double *p_zcorn_v,
                            int    *p_actnum_v,
                            int    k1,
                            int    k2,
                            int    force_close,
                            int    option,
                            int    debug
                            )
{
    int i, j, k, ic, istat, ib, np1, np2, ier=0;
    double  xg, yg, zg;
    int iflag, npoly;

    char s[24]="grd3d_inact_outside_pol";

    xtgverbose(debug);

    if (option==0) {
	xtg_speak(s,1,"Masking a grid with polygon (UNDEF outside) ...");
    }
    else{
	xtg_speak(s,1,"Masking a grid with polygon (UNDEF inside) ...");
    }

    xtg_speak(s,2,"NX NY NZ is %d %d %d", nx, ny, nz);


    for (k = k1; k <= k2; k++) {
	xtg_speak(s, 2, "Layer is %d", k);

	for (j=1; j<=ny; j++) {
	    for (i=1; i<=nx; i++) {
		grd3d_midpoint(i, j, k, nx, ny, nz, p_coord_v,
			       p_zcorn_v, &xg, &yg, &zg, debug);

		istat=0;

		ib=x_ijk2ib(i,j,k,nx,ny,nz,0);

		xtg_speak(s,3,"Midpoint is %f %f for %d %d %d",
                          xg, yg, i, j, k);

                /* check all polygons */
                /* for outside, need to make flag system so the cell is */
                /* outside all polys that are spesified */

                iflag = 0;

                np1 = 0;
                np2 = 0;
                npoly = 0;
                for (ic=0; ic < npx; ic++) {
                    if (p_xp_v[ic] == 999.0) {
                        np2 = ic - 1;
                        if (np2 > np1 + 2) {
                            xtg_speak(s, 2, "IC = %d  NP1 NP2 %d %d",
                                      ic, np1, np2);
                            xtg_speak(s, 2, "X at NP1 NP2 %f %f",
                                      p_xp_v[np1], p_xp_v[np2]);


                            istat = polys_chk_point_inside(xg, yg, p_xp_v,
                                                           p_yp_v,
                                                           np1, np2, debug);

                            if (istat < 0) {
                                /* some problems, .eg. poly is not closed */
                                xtg_warn(s, 2, "A polygon is not closed");
                                ier = 1;
                            }
                            else{
                                if (option==0 && istat>0 ) {
                                    iflag = 1;
                                }

                                else if (option==1 && istat==0 ) {
                                    iflag ++;
                                }
                                npoly ++;
                            }
                        }
                        np1 = ic + 1;
                    }
                }

                xtg_speak(s, 2, "NPOLY and IFLAG  %d  %d", npoly, iflag);

                if (option == 0 && iflag == 1) {
                    p_actnum_v[ib]=0;
                }
                if (option == 1 && iflag > 0 && iflag==npoly) {
                    p_actnum_v[ib]=0;
                }
            }
        }
    }
    return ier;
}


/* LEGACY for Perl */

void grd3d_inact_outs_pol(
			  int    np,
			  double  *p_xp_v,
			  double  *p_yp_v,
			  int    nx,
			  int    ny,
			  int    nz,
			  double  *p_coord_v,
			  double  *p_zcorn_v,
			  int    *p_actnum_v,
			  int    *p_subgrd_v,
			  int    isub,
			  int    nsub,
			  int    option,
			  int    debug
			  )
{
    int i, j, k, kc, np1, istat, ib, nz1, nz2, nz_true;
    double  xg, yg, zg;

    double x, y;
    char s[24]="grd3d_inact_outs_pol";

    xtgverbose(debug);

    if (option==0) {
	xtg_speak(s,1,"Masking a map with polygon (UNDEF outside) ...");
    }
    else{
	xtg_speak(s,1,"Masking a map with polygon (UNDEF inside) ...");
    }

    xtg_speak(s,2,"NX NY NZ is %d %d %d", nx, ny, nz);

    /*
     *-------------------------------------------------------------------------
     * It is possible to work with just one subgrid (
     * Must do some calculations for this here:
     *-------------------------------------------------------------------------
     */
    nz_true = nz;  //!!! NOTE some is unresolved regarding the use of nz_true!!
    nz1 = 1;
    nz2 = nz;
    xtg_speak(s, 2, "NZ_true", nz_true);

    if (isub <= nsub) {
        if (isub > 0) {
	    /* redefine nz */
	    nz_true = p_subgrd_v[isub-1]; /*vector starts at 0 */

	    /* find nz1 and nz2 (counted from top) */
	    k=0;
	    for (kc=0;kc<(isub-1);kc++) {
		k= k + p_subgrd_v[kc];
	    }
	    nz1=k+1;
	    nz2=k+p_subgrd_v[isub-1];
	    xtg_speak(s,2,"Using subgrid %d, using K range: %d - %d",isub,nz1,nz2);
	}
    }
    else{
	xtg_error(s,"Fatal error: isub too large");
    }



    for (k=nz1;k<=nz2;k++) {
	xtg_speak(s,2,"Layer is %d",k);
	for (j=1;j<=ny;j++) {
	    for (i=1;i<=nx;i++) {
		grd3d_midpoint(i,j,k,nx,ny,nz,p_coord_v,
			       p_zcorn_v,&xg,&yg,&zg,debug);


		x= xg;
		y= yg;
		/* search if XG, YG is present in polygon */
		istat = 0;
		np1 = 0;   //??? NP1 usage?
                xtg_speak(s, 3, "NP1", np1);

		ib=x_ijk2ib(i,j,k,nx,ny,nz,0);

		xtg_speak(s,3,"Midpoint is %f %f for %d %d %d", x,y, i, j, k);

		istat=pol_chk_point_inside(
					   x,
					   y,
					   p_xp_v,
					   p_yp_v,
					   np,
					   debug
					   );
		if (option==0 && istat==0 ) {
		    p_actnum_v[ib]=0;
		}

		if (option==1 && istat>=1 ) {
		    p_actnum_v[ib]=0;
		}

		// break;
	    }
	}
    }
}
