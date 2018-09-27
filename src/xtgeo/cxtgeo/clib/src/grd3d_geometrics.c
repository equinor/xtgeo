/*
 * ############################################################################
 * Name:      grd3d_geometrics.c
 * Author:    jriv@statoil.com
 * ############################################################################
 * Get min/max, dx,dy,dz etc of geometry
 *
 * Arguments:
 *     nx..nz           grid dimensions
 *     p_coord_v        Coordinates
 *     p_zcorn_v        ZCORN array (pointer) of input
 *     p_actnum_v       ZCORN array (pointer) of input
 *     xori...zori      Return values for XYZ origin
 *     xmin, xmax, ...  Pointers to return values for X Y Z min and max
 *     rotation         rotation in degrees fro X axis anticlockwise,
 *                      normal math way
 *     dx...dz          Average increment
 *     option1          0: use all cells; 1, only cells with ACTNUM = 1,
 *                      2: all cells for XY, active only for Z cells
 *     option2          0: compute using cell corners, 1: use cell centers
 *                      as reference (cell center always used for dx/dy/dz)
 *                      Note that option2=1 is best(?) of the grid is used
 *                      for cube or map settings
 *     debug            debug/verbose flag
 *
 * Return:
 *     The routine returns an int number stating the quality
 *     1: The grid is fairly regular, given by:
 *        - changes is dx, dy, dz is within 5%
 *        - changes in rotation is within 5%
 *     2: The grid is more irregular...
 *
 * Caveeats/issues:
 *     DX DY DZ does not match RMS's calculations fully, but I don't know why
 *
 * ############################################################################
 */
#include <math.h>
#include "libxtg.h"
#include "libxtg_.h"


int grd3d_geometrics(
		     int      nx,
		     int      ny,
		     int      nz,
		     double   *p_coord_v,
		     double   *p_zcorn_v,
		     int      *p_actnum_v,
		     double   *xori,
		     double   *yori,
		     double   *zori,
		     double   *xmin,
		     double   *xmax,
		     double   *ymin,
		     double   *ymax,
		     double   *zmin,
		     double   *zmax,
		     double    *rotation,
		     double    *dx,
		     double    *dy,
		     double    *dz,
		     int      option1,
		     int      option2,
		     int      debug
		     )

{
    /* locals */
    int     i, j, k, ib, ibn, ic, n;
    double  vxmin, vxmax, vymin, vymax, vzmin, vzmax;
    double  xv, yv, zv, sumx, sumy, sumz, sumr_x,
	vdx_ic, vdy_ic, vdz_ic, vrotx_ic;
    double  vdxmin, vdxmax, vdymin, vdymax, vdzmin, vdzmax, vxori, vyori, vzori;
    double  vrmin, vrmax;
    double  vdx, vdy, vdz, vrot, lx, ly, dum1, dum2;
    double  c[24];
    char    s[24]="grd3d_geometrics";
    int     nxuse, nyuse, nzuse, ncells, istat;

    /* some working arrays */
    double *tmp_x, *tmp_y, *tmp_z;

    /* allocation of space depends on option2 */

    nxuse = nx; nyuse = ny; nzuse = nz;

    ncells = nxuse * nyuse * nzuse;

    tmp_x = calloc(ncells, sizeof(double));
    tmp_y = calloc(ncells, sizeof(double));
    tmp_z = calloc(ncells, sizeof(double));



    xtgverbose(debug);
    xtg_speak(s,2,"Entering %s",s);
    xtg_speak(s,3,"NX NY NZ: %d %d %d", nx, ny, nz);


    /* initialize */

    /* absolutes */
    vxmin = VERYLARGEFLOAT;
    vxmax = VERYSMALLFLOAT;
    vymin = VERYLARGEFLOAT;
    vymax = VERYSMALLFLOAT;
    vzmin = VERYLARGEFLOAT;
    vzmax = VERYSMALLFLOAT;

    /* deltas */
    vdxmin = VERYLARGEFLOAT;
    vdxmax = VERYSMALLFLOAT;
    vdymin = VERYLARGEFLOAT;
    vdymax = VERYSMALLFLOAT;
    vdzmin = VERYLARGEFLOAT;
    vdzmax = VERYSMALLFLOAT;

    /* rotation */
    vrmin = VERYLARGEFLOAT;
    vrmax = VERYSMALLFLOAT;

    vdx  = 0.0;
    vdy  = 0.0;
    vdz  = 0.0;
    vrot = 0.0;

    vxori = 0.0;
    vyori = 0.0;
    vzori = 0.0;


    /* compute */

    xtg_speak(s,1,"Finding average grid geometrics...");
    xtg_speak(s,1,"Step 1 sampling geometry ...");

    for (k = 1; k <= nz; k++) {
	xtg_speak(s,3,"Finished layer %d of %d",k,nz);
	for (j = 1; j <= ny; j++) {
	    for (i = 1; i <= nx; i++) {

		ib=x_ijk2ib(i,j,k,nx,ny,nz,0);

		grd3d_midpoint(i,j,k,nx,ny,nz,p_coord_v,p_zcorn_v,
			       &xv, &yv, &zv, debug);


		tmp_x[ib]=xv;
		tmp_y[ib]=yv;
		tmp_z[ib]=zv;


		/* cell corners as reference for min/max geometry */
		if (option2 == 0) {

		    /* get the cell geometry for each cell */

		    ib=x_ijk2ib(i,j,k,nx,ny,nz,0);

		    grd3d_corners(i, j, k, nx, ny, nz,
				  p_coord_v, p_zcorn_v, c, debug);

		    /* get the origon of cell 1 1 1, corner 1 */
		    if (i==1 && j==1 && k==1) {
			vxori=c[0];
			vyori=c[1];
			vzori=c[2];
		    }

		    if ((option1==1 && p_actnum_v[ib]==1) ||
			option1==0 || option1==2) {

		    	/* the x coords */
		    	for(n=0;n<24;n=n+3) {
		    	    if ( vxmin > c[n] ) vxmin=c[n];
		    	    if ( vxmax < c[n] ) vxmax=c[n];
		    	}

		    	/* the y coords */
		    	for(n=1;n<24;n=n+3) {
		    	    if ( vymin > c[n] ) vymin=c[n];
		    	    if ( vymax < c[n] ) vymax=c[n];
		    	}

		    	/* the z coords */
			if (option1==0 || (option1>=1 && p_actnum_v[ib]==1)) {
			    for(n=2;n<24;n=n+3) {
				if ( vzmin > c[n] ) vzmin=c[n];
				if ( vzmax < c[n] ) vzmax=c[n];
			    }
			}
		    }

		}

		/* or think cell center as reference for geometry */
		else if (option2 == 1) {

		    /* get the origon of cell 1 1 1, mid-point */
		    if (i==1 && j==1 && k==1) {
			vxori=xv;
			vyori=yv;
			vzori=zv;
		    }


		    if ((option1==1 && p_actnum_v[ib]==1) ||
			option1==0 || option1==2) {

			if ( vxmin > xv ) vxmin=xv;
			if ( vxmax < xv ) vxmax=xv;

			if ( vymin > yv ) vymin=yv;
			if ( vymax < yv ) vymax=yv;

			if (option1==0 || (option1>=1 && p_actnum_v[ib]==1)) {
			    if ( vzmin > zv ) vzmin=zv;
			    if ( vzmax < zv ) vzmax=zv;
			}
		    }
		}
	    }
	}
    }


    /* do the calculations; try to make running averages, and also
       keep track of min/max values of each */

    xtg_speak(s,1,"Step 2 analyzing geometry ...");

    ic     = 1;
    sumx   = 0.0;
    sumy   = 0.0;
    sumz   = 0.0;
    sumr_x = 0.0;

    for (k = 1; k < nzuse; k++) {
	for (j = 1; j < nyuse; j++) {
	    for (i = 1; i < nxuse; i++) {

		ib  = x_ijk2ib(i,j,k,nx,ny,nz,0);

		/* ========================================= along X axis: */
		ib  = x_ijk2ib(i,j,k,nx,ny,nz,0);
		ibn = x_ijk2ib(i+1,j,k,nx,ny,nz,0);

		lx = fabs (tmp_x[ibn]-tmp_x[ib]);
		ly = fabs (tmp_y[ibn]-tmp_y[ib]);
		vdx_ic = sqrt (lx*lx + ly*ly);
		vdx = (sumx + vdx_ic)/ic;

		sumx = sumx + vdx_ic;

		if ( vdxmin > vdx_ic ) vdxmin=vdx_ic;
		if ( vdxmax < vdx_ic ) vdxmax=vdx_ic;


		/* rotation (test along x, sufficient?)*/

		x_vector_info2(tmp_x[ib], tmp_x[ibn], tmp_y[ib], tmp_y[ibn],
			       &dum1, &dum2, &vrotx_ic, 1, debug);

		/* special case if angle is close to 0 or 360: then problems
		   in averaging may occur. The solution is to avoid numbers
		   close to 360 (use cutoff 340), instead use negative angles
		*/
		if (vrotx_ic > 340) vrotx_ic = vrotx_ic - 360;

		vrot     = (sumr_x + vrotx_ic)/ic;

		sumr_x   = sumr_x + vrotx_ic;

		if ( vrmin > vrotx_ic ) vrmin=vrotx_ic;
		if ( vrmax < vrotx_ic ) vrmax=vrotx_ic;

		if (debug > 3) {
		    xtg_speak(s,4,"VROTX_IC = %f", vrotx_ic);
		    xtg_speak(s,4,"VROT = %f", vrot);
		}

		/* ========================================= along Y axis: */
		ib  = x_ijk2ib(i,j,k,nx,ny,nz,0);
		ibn = x_ijk2ib(i,j+1,k,nx,ny,nz,0);


		lx = fabs (tmp_x[ibn]-tmp_x[ib]);
		ly = fabs (tmp_y[ibn]-tmp_y[ib]);

		vdy_ic = sqrt (lx*lx + ly*ly);
		vdy = (sumy + vdy_ic)/ic;
		sumy = sumy + vdy_ic;

		if ( vdymin > vdy_ic ) vdymin=vdy_ic;
		if ( vdymax < vdy_ic ) vdymax=vdy_ic;



		/* ========================================= along Z axis: */
		ib  = x_ijk2ib(i,j,k,nx,ny,nz,0);
		ibn = x_ijk2ib(i,j,k+1,nx,ny,nz,0);

		if (p_actnum_v[ib]==1 && p_actnum_v[ibn]==1) {
		    vdz_ic = fabs (tmp_z[ibn]-tmp_z[ib]);
		    vdz = (sumz + vdz_ic)/ic;
		    sumz = sumz + vdz_ic;

		    if ( vdzmin > vdz_ic ) vdzmin=vdz_ic;
		    if ( vdzmax < vdz_ic ) vdzmax=vdz_ic;
		}

		ic++;
	    }
	}
    }
    /* deltas...*/

    xtg_speak(s,2,"SUMX=%10.2f SUMY=%10.2f  SUMZ=%10.2f IC=%d",
	      sumx, sumy, sumz, ic);


    /* checks */
    istat=1;
    if (fabs((vdxmin-vdxmax)/0.5*(vdxmin+vdxmax)) > 0.05) istat=2;
    if (fabs((vdymin-vdymax)/0.5*(vdymin+vdymax)) > 0.05) istat=2;
    if (fabs((vdzmin-vdzmax)/0.5*(vdzmin+vdzmax)) > 0.05) istat=2;
    if (fabs((vrmin-vrmax)/0.5*(vrmin+vrmax)) > 0.05) istat=2;



    xtg_speak(s,2,"Step 2 analyzing geometry ... DONE");

    *xmin = vxmin;
    *xmax = vxmax;

    *ymin = vymin;
    *ymax = vymax;

    *zmin = vzmin;
    *zmax = vzmax;

    if (vrot<0.0) vrot = vrot + 360;
    *rotation = vrot;
    *dx       = vdx;
    *dy       = vdy;
    *dz       = vdz;

    *xori     = vxori;
    *yori     = vyori;
    *zori     = vzori;


    xtg_speak(s,2,"XMIN=%10.2f XMAX=%10.2f  YMIN=%10.2f YMAX=%10.2f"
	      "  ZMIN=%10.2f ZMAX=%10.2f",
	      vxmin, vxmax, vymin, vymax, vzmin, vzmax);

    xtg_speak(s,2,"XORI=%10.2f YORI=%10.2f ZORI=%10.2f   XINC=%10.4f"
	      " YINC=%10.4f ZINC=%10.4f   ROTATION=%10.2f",
	      vxori, vyori, vzori, vdx, vdy, vdz, vrot);


    xtg_speak(s,2,"Exit from %s",s);
    return (istat);
}
