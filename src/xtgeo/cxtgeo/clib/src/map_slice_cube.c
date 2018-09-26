/*
 * ############################################################################
 * Name: map_slice_cube
 * By:   JCR
 * ############################################################################
 */


#include "libxtg.h"
#include "libxtg_.h"

/*
 * ############################################################################
 * Input a depth/time map, returns an updated map array that holds values
 * sampled from a cube
 * ############################################################################
 */

void map_slice_cube (
		     int   ncx,
		     int   ncy,
		     int   ncz,
		     double cxori,
		     double cxinc,
		     double cyori,
		     double cyinc,
		     double czori,
		     double czinc,
		     float *p_cubeval_v,
		     double crotation,
                     int yflip,
		     int   mx,
		     int   my,
		     double xmin,
		     double xstep,
		     double ymin,
		     double ystep,
		     double *p_zslice_v,   /* input map with Z values */
		     double *p_map_v,      /* the map to update */
		     int   option,
		     int   debug
		     )

{
    /* locals */
    char  s[24] = "map_slice_cube";
    int   i, j, k, ibm, im, jm, istat, ier;
    double x, y, z, rx, ry, rz;
    float value;


    xtgverbose(debug);

    xtg_speak(s,2,"Entering this routine ...");

    /* work with every map corner */
    for (jm=1;jm<=my;jm++) {
	xtg_speak(s,2,"Working with map row %d of %d ...",jm,my);
	for (im=1;im<=mx;im++) {

	    ibm=x_ijk2ib(im,jm,1,mx,my,1,0);

	    x = xmin + (im-1)*xstep;
	    y = ymin + (jm-1)*ystep;
	    z = p_zslice_v[ibm];


	    xtg_speak(s,4,"Point %8.2f %8.2f %6.2f...",x,y,z);

	    if (z < UNDEF_MAP_LIMIT) {

		istat=cube_ijk_from_xyz(
					&i,
					&j,
					&k,
					&rx,
					&ry,
					&rz,
					x,
					y,
					z,
					cxori,
					cxinc,
					cyori,
					cyinc,
					czori,
					czinc,
					ncx,
					ncy,
					ncz,
					crotation,
                                        yflip,
					0,
					debug
					);
		/* now get the value */



		if (istat != -1) {
		    ier = cube_value_ijk(
					 i,
					 j,
					 k,
					 ncx,
					 ncy,
					 ncz,
					 p_cubeval_v,
                                         &value,
					 debug
					 );

                    if (ier != EXIT_SUCCESS) {
                        xtg_error(s, "Fatal error in %s, STOP", s);
                        exit(-9);
                    }

		    p_map_v[ibm]=value;
		}
		else{
		    p_map_v[ibm]=UNDEF_MAP;
		}
	    }
	}
    }


    xtg_speak(s,2,"Exiting <map_slice_grd3d>");
}
