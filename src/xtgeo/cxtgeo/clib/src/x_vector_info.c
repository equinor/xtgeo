/*
 * ############################################################################
 * Take 2 points in XY space and compute length and azimuth or normal angle
 * Option 0, AZIMUTH is returned (clockwise, releative to North)
 * Option 1, ANGLE is returned (counter clockwise, relative to East)
 * JRIV
 * Note:
 * Angles shall be in range 0-360 degrees (no negative angles)
 * ############################################################################
 */



#include <math.h>
#include "libxtg.h"
#include "libxtg_.h"


int x_vector_info2 (
                    double  x1,
                    double  x2,
                    double  y1,
                    double  y2,
                    double  *vlen,
                    double  *xangle_radian,
                    double  *xangle_degrees,
                    int     option,
                    int     debug
                    )
{
    /* locals */
    char     sub[24]="x_vector_info2";
    double   azi, deg;


    xtgverbose(debug);
    xtg_speak(sub,3,"Entering routine");

    /*
     * ------------------------------------------------------------------------
     * Some checks
     * ------------------------------------------------------------------------
     */

    if ( x1 == x2 && y1 == y2) {
        xtg_speak(sub,3,"Hmmm null length vector");
        *vlen=0.000001;
        *xangle_radian=0.0;
        *xangle_degrees=0.0;
        return 0;
    }


    /*
     * ------------------------------------------------------------------------
     * Compute
     * ------------------------------------------------------------------------
     */


    *vlen=sqrt(powf(x2-x1,2)+powf(y2-y1,2));

    deg=0.0;

    if ((x2-x1) > 0.00001 || (x2-x1) < -0.00001) {

        deg=atan((y2-y1)/(x2-x1));
        /* western quadrant */
        if (x2 > x1) {
            azi=PI/2-deg;
        }
        /* eastern quadrant */
        else {
            deg=deg+PI;
            azi=2*PI+PI/2-deg;
        }

    }
    else{
        if (y2<y1) {
            azi=PI;
            deg=-PI/2.0;
        }
        else{
            azi=0;
            deg=PI/2;
        }
    }

    if (azi<0) azi=azi+2*PI;
    if (azi>2*PI) azi=azi-2*PI;

    if (deg<0) deg=deg+2*PI;
    if (deg>2*PI) deg=deg-2*PI;


    *xangle_radian=azi;

    /* normal school angle */
    if (option == 1) {
        *xangle_radian = deg;
        xtg_speak(sub,3,"Mode is counter-clockwise angle relative to East");
    }
    else{
        xtg_speak(sub,3,"Mode is clockwise azimuth relative to North");
    }

    *xangle_degrees=*(xangle_radian)*180/PI;


    xtg_speak(sub,3,"Y1 Y2 X1 X2: %6.2f %6.2f %6.2f %6.2f", y1, y2, x1, x2);
    xtg_speak(sub,3,"AZI DEG = %6.2f (radian %6.3f) and LEN = %6.2f",
              *xangle_degrees, *xangle_radian, *vlen);

    return 1;
}

/* ============================================================================
 * Vector length in 3D
 */


double x_vector_len3d (
                       double  x1,
                       double  x2,
                       double  y1,
                       double  y2,
                       double  z1,
                       double  z2
                       )
{
    double vlen;


    if (x1 == x2 && y1 == y2 && z1 == z2) {
        vlen=10E-20;
    }

    vlen = sqrt(powf(x2 - x1, 2) + powf(y2 - y1, 2) + powf(z2 - z1, 2));

    return vlen;
}


/*
 * ############################################################################
 * To come...
 * 3D version x_vector_info3
 * JRIV
 * ############################################################################
 */
