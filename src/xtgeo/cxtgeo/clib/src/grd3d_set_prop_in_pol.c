/*
 * ############################################################################
 * grd3d_set_prop_in_pol.c
 * Assign cell prop inside or outside a given polygon
 * Author: J.C. Rivenaes
 * ############################################################################
 */

#include "libxtg.h"
#include "libxtg_.h"

void grd3d_set_prop_in_pol(
			   int    np,
			   double  *p_xp_v,
			   double  *p_yp_v,
			   int    nx,
			   int    ny,
			   int    nz,
			   double  *p_coord_v,
			   double  *p_zcorn_v,
			   int    *p_actnum_v,
			   int    *p_prop_v,
			   double  value,
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
    int i, j, k,  ib, np1,  istat;
    double  xg, yg, zg;

    double x, y;
    char s[24]="grd3d_set_prop_in_pol";

    xtgverbose(debug);

    xtg_speak(s,2,"Assigning a prop within polygon (as-is outside) ...");
    xtg_speak(s,2,"NX NY NZ is %d %d %d", nx, ny, nz);

    if (i1>nx || i2>nx || i1<1 || i2<1 || i1>i2) {
	xtg_warn(s, 1, "I1 I2 NX: %d %d %d", i1, i2, nx);
	xtg_error(s,"Error in I specification. STOP");
    }
    if (j1>ny || j2>ny || j1<1 || j2<1 || j1>j2) {
	xtg_warn(s, 1, "J1 J2 NY: %d %d %d", j1, j2, ny);
	xtg_error(s,"Error in J specification. STOP");
    }
    if (k1>nz || k2>nz || k1<1 || k2<1 || k1>k2) {
        xtg_warn(s, 1, "K1 K2 NZ: %d %d %d", k1, k2, nz);
	xtg_error(s,"Error in K specification. STOP");
    }

    for (k=k1;k<=k2;k++) {
	xtg_speak(s,2,"Layer is %d",k);
	for (j=j1;j<=j2;j++) {
	    for (i=i1;i<=i2;i++) {
		grd3d_midpoint(i,j,k,nx,ny,nz,p_coord_v,
			       p_zcorn_v,&xg,&yg,&zg,debug);


		x= xg;
		y= yg;
		/* search if XG, YG is present in polygon */
		istat=0;
		np1=0;

		ib=x_ijk2ib(i,j,k,nx,ny,nz,0);

		xtg_speak(s,3,"Midpoint is %f %f for %d %d %d", x,y, i, j, k);

		/* the following will return 2 if point is inside, 1 on edge,
                   and 0 outside. -1 if undetermenined */
		istat=pol_chk_point_inside(
					   x,
					   y,
					   p_xp_v,
					   p_yp_v,
					   np,
					   debug
					   );

                if (option == 0) {
                    if (istat>0) {
			p_prop_v[ib]=value;
		    }
                }
                else if (option == 1) {
                    if (istat==0) {
			p_prop_v[ib]=value;
		    }
                }
		else{
                    xtg_error(s, "Invalid mode (bug) in %s", s);
                }
            }
	}
    }
}
