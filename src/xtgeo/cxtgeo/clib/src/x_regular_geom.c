/*
 * ############################################################################
 *
 * NOT FINSIHED! NOT TESTED. CHECK ANGLES!
 *
 *
 * x_regular_geom.c
 *
 * Routine(s) computing some simple geometrical stuff. If find the X, Y, Z
 * of cell I,J and the corners of the surrounding imaginary box. Rotation
 * is in ? For regular geometries only.
 *
 * If k=0, this is a map
 *
 *
 * The points should be organized as follows:
 *
 *     3       4          N
 *         *              |
 *     1       2          |___E
 *
 * lower (unless a map):
 *
 *     7       8          N
 *         *              |
 *     5       6          |___E  (Z is positve down)
 *
 *
 * The 8 corners of the cube is in a 24 index double array:
 * (x1,y1,z1,x2,y2,z2, ... x8,y8,z8)
 *
 * ############################################################################
 * ToDo:
 * -
 * ############################################################################
 * ############################################################################
 */



#include <math.h>
#include "libxtg.h"
#include "libxtg_.h"


void x_regular_geom (
		     double xmin,
		     double xinc,
		     double ymin,
		     double yinc,
		     double zmin,
		     double zinc,
		     int   nx,
		     int   ny,
		     int   nz,
		     int   i,
		     int   j,
		     int   k,
		     double rot_azi_deg,
		     double *xcenter,
		     double *ycenter,
		     double *zcenter,
		     double *corners_v,
		     int   flag,
		     int   debug
		     )
{
    /* locals */
    char      sub[24]="x_regular_geom";
    double    dist, angle;


    xtgverbose(debug);
    xtg_speak(sub,3,"Entering routine");

    /*
     * ########################################################################
     * Some checks
     * ########################################################################
     */
    if ( i < 1 || 1 > nx || j < 1 || j > ny || k < 0 || k > nz) {
	xtg_error(sub,"Invalid input; i, j or k out of bound");
    }

    angle=(rot_azi_deg-90)*PI/180.0;  /* angle known from school, in radians */


    /* distance from origo */
    dist=sqrt((i-1)*xinc + (j-1)*yinc);
    *xcenter=dist*cos(angle);
    *ycenter=dist*sin(angle);

    if (k>0) {
	*zcenter=zmin+(k-0.5)*zinc;
    }
    else{
	*zcenter=UNDEF;
    }

    /* find the 4 corners and store in array */

    /* corner 1 and 5 */
    dist=sqrt((i-1.5)*xinc + (j-1.5)*yinc);
    corners_v[0]=dist*cos(angle);
    corners_v[1]=dist*sin(angle);
    corners_v[2]=zmin+(k-1)*zinc;

    if (k>0) {
	corners_v[12]=dist*cos(angle);
	corners_v[13]=dist*sin(angle);
	corners_v[14]=zmin+k*zinc;
    }

    /* corner 2 and 6 */
    dist=sqrt((i-0.5)*xinc + (j-1.5)*yinc);
    corners_v[3]=dist*cos(angle);
    corners_v[4]=dist*sin(angle);
    corners_v[5]=zmin+(k-1)*zinc;

    if (k>0) {
	corners_v[15]=dist*cos(angle);
	corners_v[16]=dist*sin(angle);
	corners_v[17]=zmin+k*zinc;
    }

    /* corner 3 and 7 */
    dist=sqrt((i-1.5)*xinc + (j-0.5)*yinc);
    corners_v[6]=dist*cos(angle);
    corners_v[7]=dist*sin(angle);
    corners_v[8]=zmin+(k-1)*zinc;

    if (k>0) {
	corners_v[18]=dist*cos(angle);
	corners_v[19]=dist*sin(angle);
	corners_v[20]=zmin+k*zinc;
    }

    /* corner 4 and 8 */
    dist=sqrt((i-0.5)*xinc + (j-0.5)*yinc);
    corners_v[9]=dist*cos(angle);
    corners_v[10]=dist*sin(angle);
    corners_v[11]=zmin+(k-1)*zinc;

    if (k>0) {
	corners_v[21]=dist*cos(angle);
	corners_v[22]=dist*sin(angle);
	corners_v[23]=zmin+k*zinc;
    }
}
