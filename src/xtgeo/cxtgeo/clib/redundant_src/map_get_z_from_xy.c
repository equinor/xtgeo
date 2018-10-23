/*
 * ############################################################################
 * map_get_z_from_xy.c
 * Given a map and a x,y point, the map Z value is returned. This is an 
 * amalgamation of the earlier map_get_corners_xy.c and x_interp_nodes
 * 
 *   |-------|
 *   | *     |
 * ->|_______|
 *   ^
 *   |
 *
 * The points should be organized as follows (nonrotated maps):
 *
 *     2       3          N
 *                        |
 *     0       1          |___E
 *
 *
 * Author: J.C. Rivenaes
 * ############################################################################
 *
 * ############################################################################
 * TODO:
 * - checking the handling of undef nodes; shall return UNDEF
 * ############################################################################
 *
 */

#include "libxtg.h"
#include "libxtg_.h"

double map_get_z_from_xy(
			double x,
			double y,

			int nx,
			int ny,
			double xstep, 
			double ystep,
			double xmin, 
			double ymin, 
			double *p_map_v,
			
			int debug
			)
{
    int    ic, jc, kc, ib=-9, ibc;
    char sub[24]="map_get_z_from_xy";
    double x_v[4], y_v[4], z_v[4];
    double z;

    xtgverbose(debug);

    xtg_speak(sub,4,"Entering routine");

    xtg_speak(sub,4,"x y %f %f ",x,y);
    xtg_speak(sub,4,"xmin ymin xstep ystep nx ny %f %f %f %f %d %d",xmin,ymin,xstep,ystep,nx,ny);

    ib=-1;

    // get ib for lower left corner

    ib=map_get_corners_xy(x,y,nx,ny,xstep,ystep,xmin,ymin,p_map_v,debug);


    // outside map, returning UNDEF value
    if (ib<0) {
	return UNDEF;
    }

    // find the values of four nodes
    
    x_ib2ijk(ib,&ic,&jc,&kc,nx,ny,1,0);


    x_v[0]=xmin+(ic-1)*xstep;
    x_v[1]=xmin+(ic)*xstep;
    x_v[2]=xmin+(ic-1)*xstep;
    x_v[3]=xmin+(ic)*xstep;

    y_v[0]=ymin+(jc-1)*ystep;
    y_v[1]=ymin+(jc-1)*ystep;
    y_v[2]=ymin+(jc)*ystep;
    y_v[3]=ymin+(jc)*ystep;


    z_v[0]=p_map_v[ib];
    ibc=x_ijk2ib(ic+1,jc,1,nx,ny,1,0);
    z_v[1]=p_map_v[ibc];
    ibc=x_ijk2ib(ic,jc+1,1,nx,ny,1,0);
    z_v[2]=p_map_v[ibc];
    ibc=x_ijk2ib(ic+1,jc+1,1,nx,ny,1,0);
    z_v[3]=p_map_v[ibc];

    // now find the Z value, using interpolation method 2 (bilinear)

    z=x_interp_map_nodes(x_v,y_v,z_v,x,y,2,debug);

    return z;

}


