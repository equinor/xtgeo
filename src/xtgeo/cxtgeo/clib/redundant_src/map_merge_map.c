/*
 * ############################################################################
 * map_merge_map.c
 *
 * Description:
 * Merge 2 maps (that have the same coords), optionally defined by a
 * single polygon
 *
 * Author: J.C. Rivenaes
 * ############################################################################
 */

#include "libxtg.h"
#include "libxtg_.h"

void map_merge_map (
		    int    nx,
		    int    ny,
		    double  xmin,
		    double  ymin,
		    double  xinc,
		    double  yinc,
		    double  *p_map1_v,
		    double  *p_map2_v,
		    double  *p_xp_v,
		    double  *p_yp_v,
		    int    np,
		    int    pflag,
		    int    debug
		    )
{

    int i, j, ib, np1, np2, istat=0;
    double x,y;
    char s[24];


    strcpy(s,"map_merge_map");

    xtgverbose(debug);
    xtg_speak(s,2,"Entering <map_merge_map>...");


    np1=0;
    np2=np-1;

    for (j=1;j<=ny;j++) {
	for (i=1;i<=nx;i++) {
	    x=xmin + (i-1)*xinc;
	    y=ymin + (j-1)*yinc;

	    /* printf("X is %6.2f and Y is %6.2f\n",x,y);  */

	    ib=x_ijk2ib(i,j,1,nx,ny,1,0);


	    if (pflag>=0) {  /* polygons present */
		// the follwing routine shall be replaced with pol_chk.... later
		istat=pol_chk_point_inside(
					   x,
					   y,
					   p_xp_v,
					   p_yp_v,
					   np,
					   debug
					   );

	    }

	    if (p_map1_v[ib] >= UNDEF_MAP_LIMIT) {

		p_map1_v[ib]=p_map2_v[ib];

		if (pflag==1) {
		    if (istat>0) {
			p_map1_v[ib]=p_map2_v[ib];
		    }
		    else{
			p_map1_v[ib]=UNDEF_MAP;
		    }
		}

		if (pflag==0) {
		    if (istat<=0) {
			p_map1_v[ib]=p_map2_v[ib];
		    }
		    else{
			p_map1_v[ib]=UNDEF_MAP;
		    }
		}
	    }
	    else{
		if (pflag==1) {
		    if (istat<1) {
			p_map1_v[ib]=UNDEF_MAP;
		    }
		}

		if (pflag==0) {
		    if (istat>0) {
			p_map1_v[ib]=UNDEF_MAP;
		    }
		}

	    }
	}
    }
}
