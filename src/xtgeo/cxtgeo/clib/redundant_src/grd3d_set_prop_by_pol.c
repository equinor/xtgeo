/*
 * #################################################################################################
 * Name:      grd3d_set_prop_by_pol.c
 * Author:    JRIV@statoil.com
 * Created:   2015-09-09
 * Updates:
 * #################################################################################################
 * Look at all points in polygon, and mark those cells that are within, seen in XY view. The 
 * algorithm is speeded up by looking in the neighbourhood after first point is found
 *
 * Arguments:
 *     np               polygon dimension
 *     p_xp_v p_yp_v    polygon array (X Y)
 *     nx..nz           grid dimensions
 *     p_coord_v        grid coords
 *     p_zcorn_v        grid zcorn
 *     p_actnum_v       grid active cell indicator
 *     option           for future use
 *     debug            debug/verbose flag
 *
 * Caveeats/issues:
 *     double and double conversion
 * #################################################################################################
 */

#include "libxtg.h"
#include "libxtg_.h"


void grd3d_set_prop_by_pol(
			   int    np,
			   double *p_xp_v,
			   double *p_yp_v,
			   int    nx,
			   int    ny,
			   int    nz,
			   double  *p_coord_v,
			   double  *p_zcorn_v,
			   int    *p_actnum_v,
			   double  *p_prop_v,
			   double  value,
			   double  ronly,
			   int    i1,
			   int    i2,
			   int    j1,
			   int    j2,
			   int    k1,
			   int    k2,
			   int    option,
			   int    debug
			   )
{
    int m, ib, ibstart, kzonly, nrad;
    double eps;
    double x, y;

    char s[24]="grd3d_set_prop_in_pol";


    eps=1e-5;

    xtgverbose(debug);

    xtg_speak(s,2,"Assigning a prop from polygon points ...");
    xtg_speak(s,2,"NX NY NZ is %d %d %d", nx, ny, nz);

    if (i1>nx || i2>nx || i1<1 || i2<1 || i1>i2) {
	xtg_error(s,"Error in I spesification. STOP");
    }

    if (j1>ny || j2>ny || j1<1 || j2<1 || j1>j2) {
	xtg_error(s,"Error in J spesification. STOP");
    }

    if (k1>nz || k2>nz || k1<1 || k2<1 || k1>k2) {
	xtg_error(s,"Error in K spesification. STOP");
    }

    kzonly=0;
    if (k1==k2) {
	kzonly=k1;
    }

    ibstart=0;

    /* check each point of the polygon */
    for (m=0;m<np;m++) {
	x = p_xp_v[m];
	y = p_yp_v[m];
	
	xtg_speak(s,2,"Point %d -->  %9.2f    %9.2f ", m, x, y);
	
	/* the following will return 2 if point is inside, 1 on edge, and 0 outside. -1 if undetermined */
	ib=grd3d_point_in_cell(ibstart, kzonly, x, y, -999.0, nx, ny, nz, 
			       p_coord_v, p_zcorn_v, p_actnum_v, 10, 1, &nrad, 
			       1, debug);
	
	if (ib>=0) {
	    if ((ronly <= -998.99) || (p_prop_v[ib]>(ronly-eps) && p_prop_v[ib]<(ronly+eps))) {
		p_prop_v[ib]=value;			    
	    }
	    ibstart=ib;
	}
    }
    xtg_speak(s,2,"Assigning a prop from polygon points ... DONE");
}



