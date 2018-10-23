/*
 * ############################################################################
 * map_pol_mask.c
 * Makes nodes outside speciefied polygons undefined.
 * Author: J.C. Rivenaes
 * ############################################################################
 * $Id: $
 * $Source: $
 *
 * $Log: $
 *
 * ############################################################################
 * General description:
 * ############################################################################
 */

#include "libxtg.h"
#include "libxtg_.h"

/*
 * ****************************************************************************
 *                          GRD2D_POLYGON_MASK
 * ****************************************************************************
 * The algorithm is to see if the map nodes lies inside some of the polygons.
 * If not, an undef value is given. If already undef, then value is kept.
 * Todo: The algorithm is straightforward and hence a bit slow...
 * ----------------------------------------------------------------------------
 *
 */
void map_pol_mask(
		  int nx,
		  int ny,
		  double xstep,
		  double ystep,
		  double xmin,
		  double ymin,
		  double *p_map_v,
		  double *p_xp_v,
		  double *p_yp_v,
		  int   np,
		  int   debug
		  )
{
    int i, j, istat, ib;
    double x, y;
    char s[24]="map_pol_mask";


    xtgverbose(debug);

    xtg_speak(s,2,"Masking a map with polygon (UNDEF outside)");

    for (j=1;j<=ny;j++) {
	for (i=1;i<=nx;i++) {
	    x=xmin + (i-1)*xstep;
	    y=ymin + (j-1)*ystep;


	    /* search if X, Y is present in the polygon */
	    istat=pol_chk_point_inside(
				       x,
				       y,
				       p_xp_v,
				       p_yp_v,
				       np,
				       debug
				       );



	    if (istat<=0) {
		ib=x_ijk2ib(i,j,1,nx,ny,1,0);
		p_map_v[ib]=UNDEF_MAP;
	    }
	}
    }
}
